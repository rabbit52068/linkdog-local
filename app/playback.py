"""Paced raw-Opus downlink playback for LinkDog WebSocket v1.

Streaming-aware: a turn is opened once with ``begin()`` (sends ``tts:start``,
emotion, and ``sentence_start``), fed incrementally with ``feed()`` as TTS
produces PCM, and closed once with ``finish()`` (drains the queue and sends
``tts:stop``). A background sender paces Opus packets at the device cadence
with a small pre-buffer to cut time-to-first-audio.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from app.device_session import DeviceSession, DeviceState
from app.tts import map_emotion, pcm_frames


SUPPORTED_EMOTIONS = {
    "neutral",
    "happy",
    "laughing",
    "funny",
    "sad",
    "angry",
    "crying",
    "loving",
    "embarrassed",
    "surprised",
    "shocked",
    "thinking",
    "winking",
    "cool",
    "relaxed",
    "delicious",
    "kissy",
    "confident",
    "sleepy",
    "silly",
    "confused",
}


class PlaybackError(RuntimeError):
    """Raised when audio cannot be encoded or delivered to the device."""


class OpusDownlinkPlayer:
    """Encode normalized PCM and send raw Opus packets at device cadence."""

    def __init__(
        self,
        session: DeviceSession,
        codec: Any,
        *,
        frame_duration_ms: int = 60,
        pre_buffer_count: int = 5,
        playback_timeout_seconds: float = 30.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if frame_duration_ms <= 0:
            raise ValueError("frame_duration_ms must be positive")
        if pre_buffer_count < 0:
            raise ValueError("pre_buffer_count must be non-negative")
        if playback_timeout_seconds <= 0:
            raise ValueError("playback_timeout_seconds must be positive")
        self.session = session
        self.codec = codec
        self.frame_duration_ms = frame_duration_ms
        self.pre_buffer_count = pre_buffer_count
        self.playback_timeout_seconds = playback_timeout_seconds
        self._sleep = sleep
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._sender_task: Optional[asyncio.Task] = None
        self._turn_active = False

    async def play(
        self,
        text: str,
        pcm: bytes,
        *,
        emotion: Optional[str] = None,
    ) -> None:
        """One-shot playback: begin, feed the whole PCM, then finish."""
        await self.begin(text, emotion=emotion)
        try:
            await self.feed(pcm)
        except PlaybackError:
            await self._recover_listening()
            raise
        await self.finish()

    async def begin(self, text: str, *, emotion: Optional[str] = None) -> None:
        """Open a speaking turn: send start, emotion, and sentence_start once."""
        selected_emotion = emotion or map_emotion(text)
        if selected_emotion not in SUPPORTED_EMOTIONS:
            selected_emotion = "neutral"

        turn_already_started = self.session.state in (
            DeviceState.THINKING,
            DeviceState.SPEAKING,
        )
        self.session.state = DeviceState.SPEAKING
        try:
            if not turn_already_started:
                await self.session.send_json({"type": "tts", "state": "start"})
            else:
                print(
                    f"[VOICE-STATE] device={self.session.device_id} "
                    "tts=start reused reason=endpoint"
                )
            await self.session.send_json({"type": "llm", "emotion": selected_emotion})
            await self.session.send_json({
                "type": "tts",
                "state": "sentence_start",
                "text": text,
            })
        except Exception as error:
            await self._recover_listening()
            raise PlaybackError(str(error)) from error

        self._turn_active = True
        self._sender_task = asyncio.create_task(self._send_loop())

    async def feed(self, pcm: bytes) -> None:
        """Enqueue PCM for playback; the background sender paces delivery."""
        if not self._turn_active:
            raise PlaybackError("feed() called before begin()")
        try:
            for frame in pcm_frames(
                pcm,
                samples_per_frame=self.codec.samples_per_channel,
                channels=self.codec.channels,
            ):
                packet = self.codec.encode(frame)
                await self._audio_queue.put(packet)
        except PlaybackError:
            raise
        except Exception as error:
            await self._recover_listening()
            raise PlaybackError(str(error)) from error

    async def finish(self) -> None:
        """Drain the queue and close the turn with a single tts:stop."""
        if not self._turn_active:
            return
        await self._audio_queue.put(None)  # sentinel
        sender = self._sender_task
        self._sender_task = None
        try:
            if sender is not None:
                try:
                    await asyncio.wait_for(
                        sender,
                        timeout=self.playback_timeout_seconds,
                    )
                except asyncio.TimeoutError as error:
                    raise PlaybackError("playback timed out") from error
        finally:
            if sender is not None and not sender.done():
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
            self._audio_queue = asyncio.Queue()
            self._turn_active = False
            await self._recover_listening()

    async def _send_loop(self) -> None:
        sent = 0
        try:
            while True:
                packet = await self._audio_queue.get()
                if packet is None:
                    return
                if sent >= self.pre_buffer_count:
                    await self._sleep(self.frame_duration_ms / 1000)
                await self.session.send_audio(packet)
                sent += 1
        except Exception as error:
            await self._recover_listening()
            raise PlaybackError(str(error)) from error

    async def _recover_listening(self) -> None:
        try:
            await self.session.send_json({"type": "tts", "state": "stop"})
        except Exception:
            pass
        self.session.state = DeviceState.LISTENING

    async def close(self) -> None:
        self.codec.close()
