"""Resident Pocket TTS backend for low-latency LinkDog speech."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Optional

import numpy as np
from scipy.signal import resample_poly

from app.tts import TTSError


def _default_model_factory() -> Any:
    from pocket_tts import TTSModel

    return TTSModel.load_model()


class PocketTTSBackend:
    """Generate resident Pocket TTS audio as 16 kHz mono s16le PCM."""

    def __init__(
        self,
        voice: str = "cosette",
        *,
        model_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.voice = voice
        self._model_factory = model_factory or _default_model_factory
        self._model: Optional[Any] = None
        self._state: Optional[Any] = None
        self._lock = threading.Lock()

    async def synthesize(self, text: str) -> bytes:
        text = text.strip()
        if not text:
            raise TTSError("TTS text is blank")
        try:
            return await asyncio.to_thread(self._synthesize_sync, text)
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError("Pocket TTS synthesis failed") from exc

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = self._model_factory()
            self._state = self._model.get_state_for_audio_prompt(self.voice)

    def _synthesize_sync(self, text: str) -> bytes:
        with self._lock:
            self._ensure_loaded()
            chunks = []
            for chunk in self._model.generate_audio_stream(
                self._state,
                text,
                copy_state=True,
            ):
                if hasattr(chunk, "detach"):
                    chunk = chunk.detach().cpu().float().numpy()
                chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))

            if not chunks:
                raise TTSError("Pocket TTS produced empty audio")
            audio = np.concatenate(chunks)
            if not audio.size:
                raise TTSError("Pocket TTS produced empty audio")

            pcm16k = resample_poly(audio, 2, 3)
            pcm16k = np.nan_to_num(pcm16k, nan=0.0, posinf=1.0, neginf=-1.0)
            pcm16k = np.clip(pcm16k, -1.0, 1.0)
            pcm = (pcm16k * 32767.0).astype("<i2").tobytes()
            if not pcm:
                raise TTSError("Pocket TTS produced empty PCM audio")
            return pcm
