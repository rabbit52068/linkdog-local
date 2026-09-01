"""Utterance-based speech recognition backends for LinkDog voice input."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Optional

import numpy as np


class ASRError(RuntimeError):
    """Raised when an utterance cannot be transcribed."""


class ASRTimeoutError(ASRError):
    """Raised when ASR exceeds the configured per-utterance deadline."""


def _default_model_factory(model_name: str, device: str, compute_type: str) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device=device, compute_type=compute_type)


class FasterWhisperASR:
    """Lazy-loaded faster-whisper backend for 16 kHz signed 16-bit mono PCM."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = "zh",
        timeout_seconds: float = 15.0,
        initial_prompt: Optional[str] = None,
        model_factory: Optional[Callable[[str, str, str], Any]] = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("ASR timeout must be positive")
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.timeout_seconds = timeout_seconds
        self.initial_prompt = initial_prompt
        self._model_factory = model_factory or _default_model_factory
        self._model: Optional[Any] = None
        self._model_lock = threading.Lock()
        self._transcribe_lock = threading.Lock()

    async def transcribe(self, pcm: bytes, sample_rate: int = 16_000) -> str:
        self._validate_pcm(pcm, sample_rate)
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32_768.0
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._transcribe_sync, audio),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise ASRTimeoutError(
                f"ASR timed out after {self.timeout_seconds:g} seconds"
            ) from exc
        except ASRError:
            raise
        except Exception as exc:
            raise ASRError("ASR transcription failed") from exc

    @staticmethod
    def _validate_pcm(pcm: bytes, sample_rate: int) -> None:
        if sample_rate != 16_000:
            raise ASRError("ASR input must be 16 kHz mono PCM")
        if not pcm:
            raise ASRError("ASR input must be non-empty")
        if len(pcm) % 2:
            raise ASRError("ASR input must contain complete signed 16-bit samples")

    def _get_model(self) -> Any:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = self._model_factory(
                        self.model_name,
                        self.device,
                        self.compute_type,
                    )
        return self._model

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        model = self._get_model()
        options = {
            "language": self.language,
            "task": "transcribe",
            "beam_size": 5,
            "vad_filter": False,
            "condition_on_previous_text": False,
            "no_speech_threshold": 0.6,
            "log_prob_threshold": None,
        }
        if self.initial_prompt:
            options["initial_prompt"] = self.initial_prompt

        with self._transcribe_lock:
            segments, _info = model.transcribe(audio, **options)
            parts = [segment.text.strip() for segment in segments if segment.text.strip()]
        return " ".join(parts)
