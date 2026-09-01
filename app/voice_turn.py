"""Voice-turn orchestration between endpointed audio and downstream agents."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Awaitable, Callable, Optional

from app.asr import ASRError, ASRTimeoutError
from app.device_session import DeviceSession, DeviceState
from app.hermes_client import HermesAPIError, HermesToolCall
from app.playback import PlaybackError
from app.sentence_splitter import SentenceSplitter
from app.tts import TTSError, extract_emotion_prefix, sanitize_spoken_text


class VoiceActionError(RuntimeError):
    """An allow-listed voice action could not be completed safely."""


REST_MESSAGE_FALLBACK = "If there's nothing else, I'll rest now."

_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U0000FE00-\U0000FE0F\U00002190-\U00002BFF]"
)


def _sanitize_rest_message(text: str) -> str:
    """Strip emoji and barks, then keep only the first short sentence."""
    text = _EMOJI_RE.sub("", text)
    text = re.sub(
        r"\b(?:woof|arf)\b[\s!.,;:?\"'-]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.split(r"(?<=[.!?])\s+", text.strip())[0]
    return text.strip()


class VoiceTurnWorker:
    """Transcribe, answer, speak, and cancel one voice turn at a time."""

    def __init__(
        self,
        session: DeviceSession,
        voice_input: Any,
        asr: Any,
        hermes: Any = None,
        tts: Any = None,
        player: Any = None,
        action_executor: Optional[Callable[[str], Awaitable[str]]] = None,
        volume_executor: Optional[Callable[[dict], Awaitable[str]]] = None,
        disconnect: Optional[Callable[[], Awaitable[None]]] = None,
        abort_cooldown_seconds: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if (tts is None) != (player is None):
            raise ValueError("tts and player must be configured together")
        self.session = session
        self.voice_input = voice_input
        self.asr = asr
        self.hermes = hermes
        self.tts = tts
        self.player = player
        self.action_executor = action_executor
        self.volume_executor = volume_executor
        self.disconnect = disconnect
        self.abort_cooldown_seconds = max(0.0, abort_cooldown_seconds)
        self._clock = clock
        self._sleep = sleep
        if hermes is not None:
            self.session.add_close_callback(hermes.close)
        self.transcripts: asyncio.Queue = asyncio.Queue(maxsize=1)
        self.responses: asyncio.Queue = asyncio.Queue(maxsize=1)
        self.asr_failures = 0
        self.asr_timeouts = 0
        self.blank_transcripts = 0
        self.hermes_failures = 0
        self.action_failures = 0
        self.tts_failures = 0
        self.playback_failures = 0
        self.aborts = 0
        self._generation = 0
        self._active_turn: Optional[asyncio.Task] = None
        self._aborted_task: Optional[asyncio.Task] = None
        self._rest_task: Optional[asyncio.Task] = None
        self._cooldown_until = 0.0
        self._idle = asyncio.Event()

    async def next_transcript(self) -> str:
        return await self.transcripts.get()

    async def next_response(self) -> str:
        return await self.responses.get()

    async def wait_until_idle(self) -> None:
        await self._idle.wait()

    async def handle_idle_timeout(self) -> None:
        """Speak a one-off rest message, then close the device connection."""
        self._rest_task = asyncio.current_task()
        try:
            message = REST_MESSAGE_FALLBACK
            if self.hermes is not None:
                try:
                    message = await self.hermes.generate_rest_message(
                        self.session.device_id
                    )
                except HermesAPIError:
                    self.hermes_failures += 1

            spoken = _sanitize_rest_message(message)
            if not spoken:
                spoken = REST_MESSAGE_FALLBACK

            if self.tts is not None:
                try:
                    pcm = await self.tts.synthesize(spoken)
                    await self.player.play(spoken, pcm)
                except TTSError:
                    self.tts_failures += 1
                except PlaybackError:
                    self.playback_failures += 1
        except asyncio.CancelledError:
            raise
        finally:
            if self._rest_task is asyncio.current_task():
                self._rest_task = None
            if (
                self.disconnect is not None
                and not asyncio.current_task().cancelling()
            ):
                await self.disconnect()

    async def run(self) -> None:
        try:
            while True:
                utterance = await self.voice_input.next_utterance()
                self._idle.clear()
                self._generation += 1
                generation = self._generation
                task = asyncio.create_task(self._process_turn(utterance, generation))
                self._active_turn = task
                try:
                    await task
                except asyncio.CancelledError:
                    if task is not self._aborted_task:
                        raise
                finally:
                    if self._active_turn is task:
                        self._active_turn = None
                    if self._aborted_task is task:
                        self._aborted_task = None
                    self._idle.set()
        finally:
            active = self._active_turn
            if active is not None and not active.done():
                active.cancel()
                await asyncio.gather(active, return_exceptions=True)
            if self.hermes is not None:
                await self.hermes.close()

    async def abort(self, reason: str = "wake_word_detected") -> None:
        """Cancel the current generation and restore the device to Listening."""
        del reason  # Reserved for structured observability in Task 8.
        self.aborts += 1
        self._generation += 1
        rest_task = self._rest_task
        if rest_task is not None and not rest_task.done():
            rest_task.cancel()
            await asyncio.gather(rest_task, return_exceptions=True)
        await self._cancel_active_turn()
        self._clear_queue(self.responses)
        self.session.discard_queued_audio()
        await self._recover_listening()
        self._cooldown_until = max(
            self._cooldown_until,
            self._clock() + self.abort_cooldown_seconds,
        )

    async def _process_turn(self, utterance: bytes, generation: int) -> None:
        self.session.state = DeviceState.THINKING
        if await self._discard_during_cooldown(generation):
            return
        if not self._is_current(generation):
            return
        try:
            text = await self.asr.transcribe(utterance, sample_rate=16_000)
        except ASRTimeoutError as error:
            self.asr_timeouts += 1
            print(
                f"[VOICE-ASR] device={self.session.device_id} "
                f"status=timeout error={str(error)!r}"
            )
            await self._recover_if_current(generation)
            return
        except ASRError as error:
            self.asr_failures += 1
            print(
                f"[VOICE-ASR] device={self.session.device_id} "
                f"status=error error={str(error)!r}"
            )
            await self._recover_if_current(generation)
            return
        if not self._is_current(generation):
            return

        text = text.strip()
        if not text:
            self.blank_transcripts += 1
            print(
                f"[VOICE-ASR] device={self.session.device_id} status=blank"
            )
            await self._recover_if_current(generation)
            return

        print(
            f"[VOICE-ASR] device={self.session.device_id} "
            f"status=ok transcript={text!r}"
        )
        await self.session.send_json({"type": "stt", "text": text})
        if not self._is_current(generation):
            return
        self._put_queue(self.transcripts, text)

        if self.hermes is None:
            return

        if hasattr(self.hermes, "stream_complete"):
            await self._process_streaming_turn(text, generation)
        else:
            await self._process_legacy_turn(text, generation)

    async def _process_legacy_turn(self, text: str, generation: int) -> None:
        """Non-streaming path: full reply, then one-shot TTS + playback."""
        try:
            response = await self.hermes.complete(self.session.device_id, text)
        except HermesAPIError:
            self.hermes_failures += 1
            await self._recover_if_current(generation)
            return
        if isinstance(response, HermesToolCall):
            response = await self._execute_tool_call(response, generation)
            if response is None:
                return
        if not self._is_current(generation):
            return

        if self.tts is not None:
            spoken_response, response_emotion = extract_emotion_prefix(response)
            spoken_response = sanitize_spoken_text(spoken_response)
            try:
                pcm = await self.tts.synthesize(spoken_response)
            except TTSError:
                self.tts_failures += 1
                await self._recover_if_current(generation)
                return
            if not self._is_current(generation):
                return
            try:
                if response_emotion is None:
                    await self.player.play(spoken_response, pcm)
                else:
                    await self.player.play(
                        spoken_response,
                        pcm,
                        emotion=response_emotion,
                    )
            except PlaybackError:
                self.playback_failures += 1
                if self._is_current(generation):
                    self.session.state = DeviceState.LISTENING
                return
            if not self._is_current(generation):
                return
        self._put_queue(self.responses, response)

    async def _process_streaming_turn(self, text: str, generation: int) -> None:
        """Streaming path: sentence pipeline, generate-and-play concurrently."""
        splitter = SentenceSplitter()
        response_parts: list[str] = []
        turn_opened = False

        try:
            async for delta in self.hermes.stream_complete(
                self.session.device_id, text
            ):
                if not self._is_current(generation):
                    return
                if isinstance(delta, HermesToolCall):
                    response = await self._execute_tool_call(delta, generation)
                    if response is None:
                        return
                    if not self._is_current(generation):
                        return
                    await self._speak_one_shot(response, generation)
                    self._put_queue(self.responses, response)
                    return

                response_parts.append(delta)
                for sentence in splitter.feed(delta):
                    if not self._is_current(generation):
                        return
                    await self._speak_sentence(sentence, generation, turn_opened)
                    turn_opened = True
        except HermesAPIError:
            self.hermes_failures += 1
            await self._recover_if_current(generation)
            return

        if not self._is_current(generation):
            return
        remaining = splitter.flush()
        if remaining:
            await self._speak_sentence(remaining, generation, turn_opened)
            turn_opened = True

        if turn_opened:
            try:
                await self.player.finish()
            except PlaybackError:
                self.playback_failures += 1
                if self._is_current(generation):
                    self.session.state = DeviceState.LISTENING
                return

        full_response = "".join(response_parts).strip()
        if full_response:
            self._put_queue(self.responses, full_response)

    async def _speak_sentence(
        self, sentence: str, generation: int, turn_opened: bool
    ) -> None:
        """Synthesize and enqueue one sentence; open the turn on the first."""
        spoken, emotion = extract_emotion_prefix(sentence)
        spoken = sanitize_spoken_text(spoken)
        if not spoken:
            return
        try:
            pcm = await self.tts.synthesize(spoken)
        except TTSError:
            self.tts_failures += 1
            await self._recover_if_current(generation)
            return
        if not self._is_current(generation):
            return
        try:
            if not turn_opened:
                await self.player.begin(spoken, emotion=emotion)
            await self.player.feed(pcm)
        except PlaybackError:
            self.playback_failures += 1
            if self._is_current(generation):
                self.session.state = DeviceState.LISTENING
            return

    async def _speak_one_shot(self, response: str, generation: int) -> None:
        """Speak a full response (tool-call confirmation) in one shot."""
        if self.tts is None:
            return
        spoken_response, response_emotion = extract_emotion_prefix(response)
        spoken_response = sanitize_spoken_text(spoken_response)
        try:
            pcm = await self.tts.synthesize(spoken_response)
        except TTSError:
            self.tts_failures += 1
            await self._recover_if_current(generation)
            return
        if not self._is_current(generation):
            return
        try:
            if response_emotion is None:
                await self.player.play(spoken_response, pcm)
            else:
                await self.player.play(
                    spoken_response, pcm, emotion=response_emotion
                )
        except PlaybackError:
            self.playback_failures += 1
            if self._is_current(generation):
                self.session.state = DeviceState.LISTENING
            return

    async def _execute_tool_call(
        self, tool_call: HermesToolCall, generation: int
    ) -> Optional[str]:
        """Execute a validated tool call and return the spoken confirmation."""
        if tool_call.name == "linkdog_action":
            executor = self.action_executor
            executor_argument = tool_call.arguments["action"]
        elif tool_call.name == "linkdog_volume":
            executor = self.volume_executor
            executor_argument = tool_call.arguments
        else:
            executor = None
            executor_argument = None
        if executor is None:
            self.hermes_failures += 1
            await self._recover_if_current(generation)
            return None
        try:
            return await executor(executor_argument)
        except VoiceActionError:
            self.action_failures += 1
            return "That didn't work, try again."

    def _is_current(self, generation: int) -> bool:
        return generation == self._generation and not self.session.closed

    async def _cancel_active_turn(self) -> None:
        task = self._active_turn
        if task is not None and not task.done():
            self._aborted_task = task
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _discard_during_cooldown(self, generation: int) -> bool:
        remaining = self._cooldown_until - self._clock()
        if remaining > 0 and self._is_current(generation):
            await self._sleep(remaining)
            if self._is_current(generation):
                self.session.state = DeviceState.LISTENING
                self.voice_input.start_listening()
            return True
        return False

    async def _recover_if_current(self, generation: int) -> None:
        if self._is_current(generation):
            await self._recover_listening()

    async def _recover_listening(self) -> None:
        print(
            f"[VOICE-STATE] device={self.session.device_id} "
            "tts=stop reason=recover"
        )
        await self.session.send_json({"type": "tts", "state": "stop"})
        self.session.state = DeviceState.LISTENING

    @staticmethod
    def _put_queue(queue: asyncio.Queue, text: str) -> None:
        VoiceTurnWorker._clear_queue(queue)
        queue.put_nowait(text)

    @staticmethod
    def _clear_queue(queue: asyncio.Queue) -> None:
        while not queue.empty():
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                break
