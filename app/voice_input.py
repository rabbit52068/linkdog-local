"""Non-blocking inbound LinkDog audio pipeline through utterance endpointing."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from app.audio_codec import OpusCodecError
from app.device_session import DeviceSession, DeviceState


class VoiceInputPipeline:
    """Decode queued Opus packets and emit endpointed PCM utterances."""

    def __init__(
        self,
        session: DeviceSession,
        codec: Any,
        endpoint: Any,
        idle_timeout_seconds: float = 0,
        on_idle_timeout: Optional[Callable[[], None]] = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if idle_timeout_seconds < 0:
            raise ValueError("idle timeout cannot be negative")
        self.session = session
        self.codec = codec
        self.endpoint = endpoint
        self.idle_timeout_seconds = idle_timeout_seconds
        self.on_idle_timeout = on_idle_timeout
        self._sleep = sleep
        self.utterances: asyncio.Queue = asyncio.Queue(maxsize=1)
        self.invalid_audio_frames = 0
        self._listening = False
        self._discard_frames = 0
        self._idle_timer_task: Optional[asyncio.Task] = None

    @property
    def is_listening(self) -> bool:
        return self._listening

    def start_listening(self) -> bool:
        if self.session.state in (DeviceState.THINKING, DeviceState.SPEAKING):
            print(
                f"[VOICE-STATE] device={self.session.device_id} "
                f"listen=start ignored turn_state={self.session.state.value}"
            )
            return False
        self.session.discard_queued_audio()
        self.endpoint.reset()
        self._clear_utterances()
        self._listening = True
        # 丟掉喚醒詞尾音（約 800ms = 13 幀 @ 60ms），避免「小斌小斌」殘音被當成指令
        self._discard_frames = 13
        self.session.state = DeviceState.LISTENING
        self._start_idle_timer()
        return True

    async def next_utterance(self) -> bytes:
        return await self.utterances.get()

    async def run(self) -> None:
        try:
            while True:
                packet = await self.session.next_audio()
                try:
                    if not self._listening:
                        continue
                    if self._discard_frames > 0:
                        self._discard_frames -= 1
                        continue
                    try:
                        pcm = self.codec.decode(packet)
                    except (OpusCodecError, ValueError):
                        self.invalid_audio_frames += 1
                        continue
                    utterance = self.endpoint.process_pcm(pcm)
                    if utterance is None:
                        continue

                    print(
                        f"[VOICE-ENDPOINT] device={self.session.device_id} "
                        f"pcm_bytes={len(utterance)}"
                    )
                    self._stop_idle_timer()
                    self._listening = False
                    self.session.state = DeviceState.THINKING
                    print(
                        f"[VOICE-STATE] device={self.session.device_id} "
                        "tts=start reason=endpoint"
                    )
                    await self.session.send_json({"type": "tts", "state": "start"})
                    self._put_utterance(utterance)
                finally:
                    self.session.audio_queue.task_done()
        finally:
            self._stop_idle_timer()
            self.codec.close()

    def _start_idle_timer(self) -> None:
        self._stop_idle_timer()
        if self.idle_timeout_seconds <= 0 or self.on_idle_timeout is None:
            return
        self._idle_timer_task = asyncio.create_task(self._run_idle_timer())

    async def _run_idle_timer(self) -> None:
        try:
            await self._sleep(self.idle_timeout_seconds)
            if not self._listening or self.session.closed:
                return
            self._listening = False
            self.session.state = DeviceState.THINKING
            self.on_idle_timeout()
        except asyncio.CancelledError:
            pass
        finally:
            if self._idle_timer_task is asyncio.current_task():
                self._idle_timer_task = None

    def _stop_idle_timer(self) -> None:
        task = self._idle_timer_task
        self._idle_timer_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    def _put_utterance(self, utterance: bytes) -> None:
        if self.utterances.full():
            try:
                self.utterances.get_nowait()
                self.utterances.task_done()
            except asyncio.QueueEmpty:
                pass
        self.utterances.put_nowait(utterance)

    def _clear_utterances(self) -> None:
        while not self.utterances.empty():
            try:
                self.utterances.get_nowait()
                self.utterances.task_done()
            except asyncio.QueueEmpty:
                break
