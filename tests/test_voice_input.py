import asyncio
import json
import unittest
from unittest.mock import patch

from app.device_session import DeviceSession, DeviceState
from app.voice_input import VoiceInputPipeline


class FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def send_text(self, text):
        self.messages.append(json.loads(text))

    async def send_bytes(self, payload):
        self.messages.append(payload)


class FakeCodec:
    def decode(self, packet):
        return b"pcm:" + packet

    def close(self):
        pass


class FakeEndpoint:
    def __init__(self):
        self.reset_count = 0
        self.frames = []

    def reset(self):
        self.reset_count += 1

    def process_pcm(self, pcm):
        self.frames.append(pcm)
        if pcm == b"pcm:end":
            return b"complete utterance"
        return None


class VoiceInputPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.websocket = FakeWebSocket()
        self.session = DeviceSession("TEST:DOG", self.websocket, audio_queue_size=32)
        self.endpoint = FakeEndpoint()
        self.pipeline = VoiceInputPipeline(
            session=self.session,
            codec=FakeCodec(),
            endpoint=self.endpoint,
        )
        self.worker = self.session.start_task(self.pipeline.run())

    async def asyncTearDown(self):
        await self.session.close()

    async def test_idle_timeout_fires_once_while_listening(self):
        delays = []
        fired = asyncio.Event()

        async def immediate_sleep(delay):
            delays.append(delay)

        def on_idle_timeout():
            fired.set()

        pipeline = VoiceInputPipeline(
            session=self.session,
            codec=FakeCodec(),
            endpoint=self.endpoint,
            idle_timeout_seconds=60,
            on_idle_timeout=on_idle_timeout,
            sleep=immediate_sleep,
        )

        pipeline.start_listening()
        await asyncio.wait_for(fired.wait(), timeout=0.2)
        await asyncio.sleep(0)

        self.assertEqual(delays, [60])
        self.assertTrue(fired.is_set())

    async def test_idle_timeout_session_close_runs_all_close_callbacks(self):
        closed = asyncio.Event()

        async def immediate_sleep(_delay):
            return None

        async def mark_closed():
            closed.set()

        self.session.add_close_callback(mark_closed)
        pipeline = VoiceInputPipeline(
            session=self.session,
            codec=FakeCodec(),
            endpoint=self.endpoint,
            idle_timeout_seconds=60,
            on_idle_timeout=lambda: self.session.start_task(self.session.close()),
            sleep=immediate_sleep,
        )
        self.session.start_task(pipeline.run())

        pipeline.start_listening()
        for _ in range(10):
            await asyncio.sleep(0)

        self.assertTrue(self.session.closed)
        self.assertTrue(closed.is_set())
        self.assertEqual(self.session._close_callbacks, [])

    async def test_ignores_audio_until_listen_start(self):
        self.session.enqueue_audio(b"ignored")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertEqual(self.endpoint.frames, [])

    async def test_listen_start_resets_endpoint(self):
        accepted = self.pipeline.start_listening()

        self.assertTrue(accepted)
        self.assertEqual(self.session.state, DeviceState.LISTENING)
        self.assertEqual(self.endpoint.reset_count, 1)

    async def test_turn_lock_rejects_listen_start_while_thinking_or_speaking(self):
        for state in (DeviceState.THINKING, DeviceState.SPEAKING):
            with self.subTest(state=state):
                self.session.state = state
                reset_count = self.endpoint.reset_count

                accepted = self.pipeline.start_listening()

                self.assertFalse(accepted)
                self.assertEqual(self.session.state, state)
                self.assertEqual(self.endpoint.reset_count, reset_count)
                self.assertFalse(self.pipeline.is_listening)

    async def test_listen_start_discards_queued_wake_word_audio(self):
        session = DeviceSession("QUEUED:DOG", FakeWebSocket(), audio_queue_size=4)
        pipeline = VoiceInputPipeline(
            session=session,
            codec=FakeCodec(),
            endpoint=FakeEndpoint(),
        )
        session.enqueue_audio(b"wake-word")

        pipeline.start_listening()

        self.assertTrue(session.audio_queue.empty())
        await session.close()

    async def test_endpoint_stops_uplink_and_emits_utterance(self):
        with patch("builtins.print") as mock_print:
            self.pipeline.start_listening()
            # 喚醒詞丟棄窗口（13 幀）會先吞掉前 13 幀，餵滿後才開始處理
            for _ in range(13):
                self.session.enqueue_audio(b"wake-tail")
            self.session.enqueue_audio(b"middle")
            self.session.enqueue_audio(b"end")

            utterance = await asyncio.wait_for(self.pipeline.next_utterance(), timeout=0.2)

        self.assertEqual(utterance, b"complete utterance")
        mock_print.assert_any_call(
            "[VOICE-ENDPOINT] device=TEST:DOG pcm_bytes=18"
        )
        mock_print.assert_any_call(
            "[VOICE-STATE] device=TEST:DOG tts=start reason=endpoint"
        )
        self.assertEqual(self.session.state, DeviceState.THINKING)
        self.assertEqual(
            self.websocket.messages,
            [{"type": "tts", "state": "start"}],
        )

    async def test_second_endpoint_is_blocked_until_next_listen_start(self):
        self.pipeline.start_listening()
        for _ in range(13):
            self.session.enqueue_audio(b"wake-tail")
        self.session.enqueue_audio(b"end")
        await asyncio.wait_for(self.pipeline.next_utterance(), timeout=0.2)
        self.session.enqueue_audio(b"end")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertTrue(self.pipeline.utterances.empty())


if __name__ == "__main__":
    unittest.main()
