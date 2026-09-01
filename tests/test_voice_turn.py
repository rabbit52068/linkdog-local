import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from app.asr import ASRError, ASRTimeoutError
from app.device_session import DeviceSession, DeviceState
from app.hermes_client import HermesToolCall, HermesUnavailableError
from app.playback import PlaybackError
from app.tts import TTSError
from app.voice_turn import VoiceActionError, VoiceTurnWorker


class FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def send_text(self, text):
        self.messages.append(json.loads(text))

    async def send_bytes(self, payload):
        self.messages.append(payload)


class FakeVoiceInput:
    def __init__(self):
        self.utterances = asyncio.Queue()
        self.start_count = 0

    async def next_utterance(self):
        return await self.utterances.get()

    def start_listening(self):
        self.start_count += 1
        return True


class FakeASR:
    def __init__(self, result="你好", error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def transcribe(self, pcm, sample_rate):
        self.calls.append((pcm, sample_rate))
        if self.error:
            raise self.error
        return self.result


class FakeHermes:
    def __init__(self, result="好的", error=None):
        self.result = result
        self.error = error
        self.calls = []
        self.closed = False

    async def complete(self, device_id, text):
        self.calls.append((device_id, text))
        if self.error:
            raise self.error
        return self.result

    async def close(self):
        self.closed = True


class FakeTTS:
    def __init__(self, pcm=b"pcm-audio", error=None):
        self.pcm = pcm
        self.error = error
        self.calls = []

    async def synthesize(self, text):
        self.calls.append(text)
        if self.error:
            raise self.error
        return self.pcm


class FakeActionExecutor:
    def __init__(self, response="Okay, I sat down.", error=None):
        self.response = response
        self.error = error
        self.calls = []

    async def __call__(self, action):
        self.calls.append(action)
        if self.error:
            raise self.error
        return self.response


class FakePlayer:
    def __init__(self, error=None):
        self.error = error
        self.calls = []
        self.emotions = []

    async def play(self, text, pcm, *, emotion=None):
        self.calls.append((text, pcm))
        self.emotions.append(emotion)
        if self.error:
            raise self.error

    async def begin(self, text, *, emotion=None):
        self.calls.append((text, b"<begin>"))
        self.emotions.append(emotion)
        if self.error:
            raise self.error

    async def feed(self, pcm):
        self.calls.append(("<feed>", pcm))
        if self.error:
            raise self.error

    async def finish(self):
        self.calls.append(("<finish>", b""))
        if self.error:
            raise self.error


class BlockingPlayer(FakePlayer):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def play(self, text, pcm):
        self.calls.append((text, pcm))
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            if asyncio.current_task().cancelling():
                self.cancelled.set()


class CancellationResistantHermes(FakeHermes):
    def __init__(self):
        super().__init__(result="過期回答")
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, device_id, text):
        self.calls.append((device_id, text))
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            await self.release.wait()
        return self.result


class StreamingHermes:
    """Fake Hermes with a stream_complete async generator."""

    def __init__(self, deltas, error=None):
        self.deltas = deltas
        self.error = error
        self.calls = []
        self.closed = False

    async def stream_complete(self, device_id, text):
        self.calls.append((device_id, text))
        if self.error:
            raise self.error
        for delta in self.deltas:
            yield delta

    async def close(self):
        self.closed = True


class VoiceTurnWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.websocket = FakeWebSocket()
        self.session = DeviceSession("TEST:DOG", self.websocket)
        self.voice_input = FakeVoiceInput()

    async def asyncTearDown(self):
        await self.session.close()

    async def start_worker(
        self,
        asr,
        hermes=None,
        tts=None,
        player=None,
        **kwargs,
    ):
        worker = VoiceTurnWorker(
            self.session,
            self.voice_input,
            asr,
            hermes=hermes,
            tts=tts,
            player=player,
            **kwargs,
        )
        self.session.start_task(worker.run())
        return worker

    async def test_idle_timeout_speaks_short_rest_message_then_disconnects(self):
        hermes = FakeHermes()
        hermes.generate_rest_message = AsyncMock(
            return_value="If there's nothing else, I'll rest now."
        )
        tts = FakeTTS(pcm=b"rest-pcm")
        player = FakePlayer()
        disconnect = AsyncMock()
        worker = VoiceTurnWorker(
            self.session,
            self.voice_input,
            FakeASR(),
            hermes=hermes,
            tts=tts,
            player=player,
            disconnect=disconnect,
        )

        await worker.handle_idle_timeout()

        hermes.generate_rest_message.assert_awaited_once_with("TEST:DOG")
        self.assertEqual(tts.calls, ["If there's nothing else, I'll rest now."])
        self.assertEqual(player.calls, [
            ("If there's nothing else, I'll rest now.", b"rest-pcm")
        ])
        disconnect.assert_awaited_once_with()

    async def test_abort_cancels_inflight_rest_flow_without_disconnect(self):
        release = asyncio.Event()

        class BlockingHermes(FakeHermes):
            async def generate_rest_message(self, device_id):
                await release.wait()
                return "rest"

        hermes = BlockingHermes()
        tts = FakeTTS()
        player = FakePlayer()
        disconnect = AsyncMock()
        worker = VoiceTurnWorker(
            self.session,
            self.voice_input,
            FakeASR(),
            hermes=hermes,
            tts=tts,
            player=player,
            disconnect=disconnect,
        )

        rest_task = asyncio.create_task(worker.handle_idle_timeout())
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        await worker.abort("wake_word_detected")
        await asyncio.gather(rest_task, return_exceptions=True)

        disconnect.assert_not_awaited()
        self.assertEqual(self.session.state, DeviceState.LISTENING)

    async def test_idle_timeout_strips_emoji_and_barks_and_keeps_one_sentence(self):
        hermes = FakeHermes()
        hermes.generate_rest_message = AsyncMock(
            return_value="🐶 Woof! I will rest now. 💤 And another sentence."
        )
        tts = FakeTTS(pcm=b"rest-pcm")
        player = FakePlayer()
        disconnect = AsyncMock()
        worker = VoiceTurnWorker(
            self.session,
            self.voice_input,
            FakeASR(),
            hermes=hermes,
            tts=tts,
            player=player,
            disconnect=disconnect,
        )

        await worker.handle_idle_timeout()

        self.assertEqual(tts.calls, ["I will rest now."])
        self.assertEqual(player.calls, [("I will rest now.", b"rest-pcm")])
        disconnect.assert_awaited_once_with()

    async def test_transcribes_utterance_sends_stt_and_emits_transcript(self):
        asr = FakeASR(result="你好 小智")
        worker = await self.start_worker(asr)
        with patch("builtins.print") as mock_print:
            await self.voice_input.utterances.put(b"pcm")

            transcript = await asyncio.wait_for(worker.next_transcript(), timeout=0.2)

        self.assertEqual(transcript, "你好 小智")
        mock_print.assert_any_call(
            "[VOICE-ASR] device=TEST:DOG status=ok transcript='你好 小智'"
        )
        self.assertEqual(asr.calls, [(b"pcm", 16_000)])
        self.assertEqual(
            self.websocket.messages,
            [{"type": "stt", "text": "你好 小智"}],
        )
        self.assertEqual(self.session.state, DeviceState.THINKING)

    async def test_sends_transcript_to_hermes_and_emits_response(self):
        hermes = FakeHermes(result="好的，我會坐下。")
        worker = await self.start_worker(FakeASR(result="請你坐下"), hermes=hermes)
        await self.voice_input.utterances.put(b"pcm")

        response = await asyncio.wait_for(worker.next_response(), timeout=0.2)

        self.assertEqual(response, "好的，我會坐下。")
        self.assertEqual(hermes.calls, [("TEST:DOG", "請你坐下")])
        self.assertEqual(
            self.websocket.messages,
            [{"type": "stt", "text": "請你坐下"}],
        )

    async def test_hermes_failure_recovers_listening_state(self):
        hermes = FakeHermes(error=HermesUnavailableError("offline"))
        worker = await self.start_worker(FakeASR(result="你好"), hermes=hermes)
        await self.voice_input.utterances.put(b"pcm")

        await asyncio.wait_for(worker.wait_until_idle(), timeout=0.2)

        self.assertEqual(worker.hermes_failures, 1)
        self.assertTrue(worker.responses.empty())
        self.assertEqual(self.websocket.messages, [
            {"type": "stt", "text": "你好"},
            {"type": "tts", "state": "stop"},
        ])
        self.assertEqual(self.session.state, DeviceState.LISTENING)

    async def test_synthesizes_and_plays_hermes_response(self):
        hermes = FakeHermes(result="好的，我會坐下。")
        tts = FakeTTS(pcm=b"normalized-pcm")
        player = FakePlayer()
        worker = await self.start_worker(
            FakeASR(result="請你坐下"),
            hermes=hermes,
            tts=tts,
            player=player,
        )
        await self.voice_input.utterances.put(b"pcm")

        response = await asyncio.wait_for(worker.next_response(), timeout=0.2)

        self.assertEqual(response, "好的，我會坐下。")
        self.assertEqual(tts.calls, ["好的，我會坐下。"])
        self.assertEqual(player.calls, [("好的，我會坐下。", b"normalized-pcm")])

    async def test_extracts_llm_emotion_before_tts_and_playback(self):
        tts = FakeTTS(pcm=b"emotion-pcm")
        player = FakePlayer()
        worker = await self.start_worker(
            FakeASR(result="hello"),
            hermes=FakeHermes(result="😘 I missed you!"),
            tts=tts,
            player=player,
        )
        await self.voice_input.utterances.put(b"pcm")

        response = await asyncio.wait_for(worker.next_response(), timeout=0.2)

        self.assertEqual(response, "😘 I missed you!")
        self.assertEqual(tts.calls, ["I missed you!"])
        self.assertEqual(player.calls, [("I missed you!", b"emotion-pcm")])
        self.assertEqual(player.emotions, ["kissy"])

    async def test_executes_validated_tool_call_before_speaking_confirmation(self):
        executor = FakeActionExecutor(response="Okay, I sat down.")
        tts = FakeTTS(pcm=b"action-pcm")
        player = FakePlayer()
        worker = await self.start_worker(
            FakeASR(result="坐下"),
            hermes=FakeHermes(result=HermesToolCall(
                "linkdog_action", {"action": "sit_down"}
            )),
            tts=tts,
            player=player,
            action_executor=executor,
        )
        await self.voice_input.utterances.put(b"pcm")

        response = await asyncio.wait_for(worker.next_response(), timeout=0.2)

        self.assertEqual(executor.calls, ["sit_down"])
        self.assertEqual(response, "Okay, I sat down.")
        self.assertEqual(tts.calls, ["Okay, I sat down."])
        self.assertEqual(player.calls, [("Okay, I sat down.", b"action-pcm")])

    async def test_action_failure_speaks_failure_without_claiming_success(self):
        executor = FakeActionExecutor(error=VoiceActionError("MCP timeout"))
        tts = FakeTTS(pcm=b"failure-pcm")
        player = FakePlayer()
        worker = await self.start_worker(
            FakeASR(result="坐下"),
            hermes=FakeHermes(result=HermesToolCall(
                "linkdog_action", {"action": "sit_down"}
            )),
            tts=tts,
            player=player,
            action_executor=executor,
        )
        await self.voice_input.utterances.put(b"pcm")

        response = await asyncio.wait_for(worker.next_response(), timeout=0.2)

        self.assertEqual(worker.action_failures, 1)
        self.assertEqual(response, "That didn't work, try again.")
        self.assertEqual(tts.calls, ["That didn't work, try again."])
        self.assertEqual(player.calls, [
            ("That didn't work, try again.", b"failure-pcm")
        ])

    async def test_executes_volume_tool_and_speaks_hardware_result(self):
        executor = FakeActionExecutor(response="Volume set to 70 percent.")
        tts = FakeTTS(pcm=b"volume-pcm")
        player = FakePlayer()
        worker = await self.start_worker(
            FakeASR(result="raise the volume"),
            hermes=FakeHermes(result=HermesToolCall(
                "linkdog_volume", {"mode": "up"}
            )),
            tts=tts,
            player=player,
            volume_executor=executor,
        )
        await self.voice_input.utterances.put(b"pcm")

        response = await asyncio.wait_for(worker.next_response(), timeout=0.2)

        self.assertEqual(executor.calls, [{"mode": "up"}])
        self.assertEqual(response, "Volume set to 70 percent.")
        self.assertEqual(tts.calls, ["Volume set to 70 percent."])
        self.assertEqual(player.calls, [
            ("Volume set to 70 percent.", b"volume-pcm")
        ])


    async def test_never_speaks_wake_word_from_hermes_response(self):
        tts = FakeTTS()
        player = FakePlayer()
        worker = await self.start_worker(
            FakeASR(result="喚醒詞是什麼"),
            hermes=FakeHermes(result="請說小斌小斌"),
            tts=tts,
            player=player,
        )
        await self.voice_input.utterances.put(b"pcm")

        await asyncio.wait_for(worker.next_response(), timeout=0.2)

        self.assertEqual(tts.calls, ["請說喚醒詞"])
        self.assertEqual(player.calls, [("請說喚醒詞", b"pcm-audio")])

    async def test_tts_failure_sends_stop_and_recovers(self):
        worker = await self.start_worker(
            FakeASR(result="你好"),
            hermes=FakeHermes(result="回答"),
            tts=FakeTTS(error=TTSError("offline")),
            player=FakePlayer(),
        )
        await self.voice_input.utterances.put(b"pcm")

        await asyncio.wait_for(worker.wait_until_idle(), timeout=0.2)

        self.assertEqual(worker.tts_failures, 1)
        self.assertEqual(self.websocket.messages[-1], {"type": "tts", "state": "stop"})
        self.assertEqual(self.session.state, DeviceState.LISTENING)

    async def test_playback_failure_is_counted_without_duplicate_recovery(self):
        worker = await self.start_worker(
            FakeASR(result="你好"),
            hermes=FakeHermes(result="回答"),
            tts=FakeTTS(),
            player=FakePlayer(error=PlaybackError("socket failed")),
        )
        await self.voice_input.utterances.put(b"pcm")

        await asyncio.wait_for(worker.wait_until_idle(), timeout=0.2)

        self.assertEqual(worker.playback_failures, 1)
        self.assertEqual(self.session.state, DeviceState.LISTENING)

    async def test_wake_word_abort_cancels_active_playback_and_sends_stop(self):
        player = BlockingPlayer()
        worker = await self.start_worker(
            FakeASR(result="你好"),
            hermes=FakeHermes(result="這是一段很長的回答"),
            tts=FakeTTS(),
            player=player,
        )
        await self.voice_input.utterances.put(b"pcm")
        await asyncio.wait_for(player.started.wait(), timeout=0.2)

        await worker.abort("wake_word_detected")

        self.assertTrue(player.cancelled.is_set())
        self.assertEqual(worker.aborts, 1)
        self.assertTrue(worker.responses.empty())
        self.assertEqual(self.websocket.messages[-1], {"type": "tts", "state": "stop"})
        self.assertEqual(self.session.state, DeviceState.LISTENING)

    async def test_late_hermes_result_after_abort_is_discarded(self):
        hermes = CancellationResistantHermes()
        tts = FakeTTS()
        worker = await self.start_worker(
            FakeASR(result="你好"),
            hermes=hermes,
            tts=tts,
            player=FakePlayer(),
        )
        await self.voice_input.utterances.put(b"pcm")
        await asyncio.wait_for(hermes.started.wait(), timeout=0.2)

        abort_task = asyncio.create_task(worker.abort("wake_word_detected"))
        await asyncio.sleep(0)
        hermes.release.set()
        await asyncio.wait_for(abort_task, timeout=0.2)

        self.assertEqual(tts.calls, [])
        self.assertTrue(worker.responses.empty())
        self.assertEqual(self.websocket.messages[-1], {"type": "tts", "state": "stop"})

    async def test_audio_endpointed_during_post_abort_cooldown_is_discarded(self):
        now = [10.0]
        delays = []

        async def fake_sleep(delay):
            delays.append(delay)
            now[0] += delay

        worker = await self.start_worker(
            FakeASR(result="下一輪"),
            abort_cooldown_seconds=2.0,
            clock=lambda: now[0],
            sleep=fake_sleep,
        )
        await worker.abort("wake_word_detected")
        await self.voice_input.utterances.put(b"pcm")

        await asyncio.wait_for(worker.wait_until_idle(), timeout=0.2)

        self.assertTrue(worker.transcripts.empty())
        self.assertEqual(delays, [2.0])
        self.assertEqual(self.voice_input.start_count, 1)

        await self.voice_input.utterances.put(b"fresh-pcm")

        transcript = await asyncio.wait_for(worker.next_transcript(), timeout=0.2)

        self.assertEqual(transcript, "下一輪")

    async def test_closes_hermes_client_when_worker_is_cancelled(self):
        hermes = FakeHermes()
        await self.start_worker(FakeASR(), hermes=hermes)

        await self.session.close()

        self.assertTrue(hermes.closed)

    async def test_streaming_turn_splits_sentences_and_plays_incrementally(self):
        hermes = StreamingHermes([
            "Hey Nelson, ",
            "I'm ready. ",
            "Let's go!",
        ])
        tts = FakeTTS(pcm=b"stream-pcm")
        player = FakePlayer()
        worker = await self.start_worker(
            FakeASR(result="hello"),
            hermes=hermes,
            tts=tts,
            player=player,
        )
        await self.voice_input.utterances.put(b"pcm")

        response = await asyncio.wait_for(worker.next_response(), timeout=0.2)

        self.assertEqual(response, "Hey Nelson, I'm ready. Let's go!")
        # First sentence cut on comma, then strong punctuation.
        self.assertEqual(tts.calls, ["Hey Nelson,", "I'm ready.", "Let's go!"])
        # begin + 3 feeds + finish.
        self.assertEqual(len(player.calls), 5)
        self.assertEqual(player.calls[0], ("Hey Nelson,", b"<begin>"))
        self.assertEqual(player.calls[-1], ("<finish>", b""))

    async def test_streaming_turn_emits_full_response(self):
        hermes = StreamingHermes(["Just one sentence."])
        worker = await self.start_worker(
            FakeASR(result="hi"),
            hermes=hermes,
            tts=FakeTTS(),
            player=FakePlayer(),
        )
        await self.voice_input.utterances.put(b"pcm")

        response = await asyncio.wait_for(worker.next_response(), timeout=0.2)

        self.assertEqual(response, "Just one sentence.")

    async def test_streaming_hermes_failure_recovers_listening(self):
        hermes = StreamingHermes([], error=HermesUnavailableError("offline"))
        worker = await self.start_worker(
            FakeASR(result="hi"),
            hermes=hermes,
            tts=FakeTTS(),
            player=FakePlayer(),
        )
        await self.voice_input.utterances.put(b"pcm")

        await asyncio.wait_for(worker.wait_until_idle(), timeout=0.2)

        self.assertEqual(worker.hermes_failures, 1)
        self.assertEqual(self.websocket.messages[-1], {"type": "tts", "state": "stop"})
        self.assertEqual(self.session.state, DeviceState.LISTENING)

    async def test_blank_transcript_recovers_listening_state(self):
        worker = await self.start_worker(FakeASR(result="   "))
        with patch("builtins.print") as mock_print:
            await self.voice_input.utterances.put(b"pcm")

            await asyncio.wait_for(worker.wait_until_idle(), timeout=0.2)

        mock_print.assert_any_call(
            "[VOICE-ASR] device=TEST:DOG status=blank"
        )
        mock_print.assert_any_call(
            "[VOICE-STATE] device=TEST:DOG tts=stop reason=recover"
        )
        self.assertEqual(
            self.websocket.messages,
            [{"type": "tts", "state": "stop"}],
        )
        self.assertEqual(self.session.state, DeviceState.LISTENING)
        self.assertTrue(worker.transcripts.empty())

    async def test_asr_error_recovers_without_emitting_transcript(self):
        worker = await self.start_worker(FakeASR(error=ASRError("broken")))
        with patch("builtins.print") as mock_print:
            await self.voice_input.utterances.put(b"pcm")

            await asyncio.wait_for(worker.wait_until_idle(), timeout=0.2)

        mock_print.assert_any_call(
            "[VOICE-ASR] device=TEST:DOG status=error error='broken'"
        )
        self.assertEqual(worker.asr_failures, 1)
        self.assertEqual(
            self.websocket.messages,
            [{"type": "tts", "state": "stop"}],
        )
        self.assertTrue(worker.transcripts.empty())

    async def test_timeout_is_counted_separately(self):
        worker = await self.start_worker(FakeASR(error=ASRTimeoutError("slow")))
        with patch("builtins.print") as mock_print:
            await self.voice_input.utterances.put(b"pcm")

            await asyncio.wait_for(worker.wait_until_idle(), timeout=0.2)

        mock_print.assert_any_call(
            "[VOICE-ASR] device=TEST:DOG status=timeout error='slow'"
        )
        self.assertEqual(worker.asr_timeouts, 1)
        self.assertEqual(worker.asr_failures, 0)


if __name__ == "__main__":
    unittest.main()
