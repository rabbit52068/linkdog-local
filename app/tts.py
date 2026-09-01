"""Text-to-speech backends and PCM framing for LinkDog playback."""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Sequence


class TTSError(RuntimeError):
    """Raised when speech synthesis or PCM normalization fails."""


EMOJI_EMOTIONS = {
    "😂": "funny",
    "😭": "crying",
    "😠": "angry",
    "😔": "sad",
    "😍": "loving",
    "😲": "surprised",
    "😱": "shocked",
    "🤔": "thinking",
    "😌": "relaxed",
    "😴": "sleepy",
    "😜": "silly",
    "🙄": "confused",
    "😶": "neutral",
    "🙂": "happy",
    "😆": "laughing",
    "😳": "embarrassed",
    "😉": "winking",
    "😎": "cool",
    "🤤": "delicious",
    "😘": "kissy",
    "😏": "confident",
}


def extract_emotion_prefix(text: str) -> tuple[str, str | None]:
    """Remove a Xiaozhi emotion emoji prefix and return its firmware emotion."""
    stripped = text.lstrip()
    if not stripped:
        return text, None
    emotion = EMOJI_EMOTIONS.get(stripped[0])
    if emotion is None:
        return text, None
    return stripped[1:].lstrip(), emotion


def sanitize_spoken_text(text: str) -> str:
    """Prevent device TTS from acoustically triggering its own wake word."""
    return text.replace("小斌小斌", "喚醒詞")


def pcm_frames(
    pcm: bytes,
    *,
    samples_per_frame: int = 960,
    channels: int = 1,
) -> Iterable[bytes]:
    """Yield fixed s16le PCM frames, zero-padding the final short frame."""
    if not pcm:
        raise TTSError("PCM audio is empty")
    if len(pcm) % 2:
        raise TTSError("PCM audio must be 16-bit aligned")
    if samples_per_frame <= 0 or channels <= 0:
        raise ValueError("frame dimensions must be positive")

    frame_bytes = samples_per_frame * channels * 2
    for offset in range(0, len(pcm), frame_bytes):
        frame = pcm[offset : offset + frame_bytes]
        if len(frame) < frame_bytes:
            frame += b"\x00" * (frame_bytes - len(frame))
        yield frame


def map_emotion(text: str) -> str:
    """Map response text to a firmware-supported deterministic emotion."""
    lowered = text.lower()
    if any(token in lowered for token in ("sorry", "failed", "cannot", "error", "sad", "unfortunately")):
        return "sad"
    if any(token in lowered for token in ("great", "good", "success", "happy", "wonderful", "!", "！")):
        return "happy"
    if "?" in text or "？" in text:
        return "thinking"
    return "neutral"


class CommandTTSBackend:
    """Use low-latency macOS say, falling back to Edge TTS, then normalize."""

    def __init__(
        self,
        *,
        voice: str = "zh-TW-HsiaoChenNeural",
        fallback_voice: str = "Meijia",
        edge_tts_command: str = "edge-tts",
        ffmpeg_command: str = "ffmpeg",
        say_command: str = "say",
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self.voice = voice
        self.fallback_voice = fallback_voice
        self.edge_tts_command = edge_tts_command
        self.ffmpeg_command = ffmpeg_command
        self.say_command = say_command
        self._runner = runner

    async def synthesize(self, text: str) -> bytes:
        text = text.strip()
        if not text:
            raise TTSError("TTS text is blank")
        return await asyncio.to_thread(self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> bytes:
        with tempfile.TemporaryDirectory(prefix="linkdog-tts-") as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "speech.aiff"
            say_result = self._run([
                self.say_command,
                "-v",
                self.fallback_voice,
                "-o",
                str(source),
                text,
            ])

            if say_result.returncode != 0:
                source = temp_path / "speech.mp3"
                edge_result = self._run([
                    self.edge_tts_command,
                    "--voice",
                    self.voice,
                    "--text",
                    text,
                    "--write-media",
                    str(source),
                ])
                if edge_result.returncode != 0:
                    raise TTSError("macOS say and Edge TTS failed")

            ffmpeg_result = self._run([
                self.ffmpeg_command,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "s16le",
                "pipe:1",
            ])
            if ffmpeg_result.returncode != 0:
                raise TTSError("ffmpeg failed to normalize TTS audio")
            pcm = bytes(ffmpeg_result.stdout or b"")
            if not pcm:
                raise TTSError("TTS produced empty PCM audio")
            if len(pcm) % 2:
                raise TTSError("TTS PCM audio is not 16-bit aligned")
            return pcm

    def _run(self, args: Sequence[str]) -> subprocess.CompletedProcess:
        try:
            return self._runner(
                list(args),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except (FileNotFoundError, OSError) as error:
            return subprocess.CompletedProcess(list(args), 127, b"", str(error).encode())
