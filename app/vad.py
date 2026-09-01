"""Server-side voice activity detection and utterance endpointing."""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Protocol


class VadConfigurationError(ValueError):
    """Raised when endpoint thresholds cannot be represented in VAD chunks."""


class VadClassifier(Protocol):
    def is_speech(self, pcm: bytes) -> bool:
        """Return whether one fixed-duration PCM chunk contains speech."""


class WebRtcVadClassifier:
    """WebRTC VAD adapter for 16-bit mono PCM chunks."""

    def __init__(self, sample_rate: int = 16_000, aggressiveness: int = 2) -> None:
        if sample_rate not in (8_000, 16_000, 32_000, 48_000):
            raise VadConfigurationError("WebRTC VAD does not support this sample rate")
        if aggressiveness not in (0, 1, 2, 3):
            raise VadConfigurationError("VAD aggressiveness must be between 0 and 3")
        try:
            import webrtcvad
        except ImportError as exc:
            raise VadConfigurationError(
                "webrtcvad is not installed; install webrtcvad-wheels"
            ) from exc
        self.sample_rate = sample_rate
        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, pcm: bytes) -> bool:
        return bool(self._vad.is_speech(pcm, self.sample_rate))


class UtteranceEndpoint:
    """Collect VAD-classified PCM chunks into one bounded utterance."""

    def __init__(
        self,
        classifier: VadClassifier,
        sample_rate: int = 16_000,
        chunk_duration_ms: int = 20,
        pre_roll_ms: int = 300,
        minimum_speech_ms: int = 300,
        end_silence_ms: int = 700,
        maximum_utterance_ms: int = 12_000,
    ) -> None:
        if sample_rate <= 0 or chunk_duration_ms not in (10, 20, 30):
            raise VadConfigurationError("VAD chunks must be 10, 20, or 30 ms")
        for name, value in (
            ("pre_roll_ms", pre_roll_ms),
            ("minimum_speech_ms", minimum_speech_ms),
            ("end_silence_ms", end_silence_ms),
            ("maximum_utterance_ms", maximum_utterance_ms),
        ):
            if value < 0 or value % chunk_duration_ms:
                raise VadConfigurationError(
                    f"{name} must be a non-negative multiple of {chunk_duration_ms} ms"
                )
        if minimum_speech_ms <= 0 or end_silence_ms <= 0 or maximum_utterance_ms <= 0:
            raise VadConfigurationError("speech, silence, and maximum thresholds must be positive")
        if minimum_speech_ms > maximum_utterance_ms:
            raise VadConfigurationError("minimum speech cannot exceed maximum utterance")

        self.classifier = classifier
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_bytes = sample_rate * chunk_duration_ms // 1000 * 2
        self._pre_roll_limit = pre_roll_ms // chunk_duration_ms
        self._minimum_speech_chunks = minimum_speech_ms // chunk_duration_ms
        self._end_silence_chunks = end_silence_ms // chunk_duration_ms
        self._maximum_utterance_chunks = maximum_utterance_ms // chunk_duration_ms
        self._pre_roll: Deque[bytes] = deque(maxlen=self._pre_roll_limit or None)
        self._utterance: List[bytes] = []
        self._speech_chunks = 0
        self._silence_chunks = 0
        self.speech_active = False

    def process_pcm(self, pcm: bytes) -> Optional[bytes]:
        if not pcm or len(pcm) % self.chunk_bytes:
            raise ValueError(
                f"PCM input must contain whole {self.chunk_duration_ms} ms chunks "
                f"({self.chunk_bytes} bytes each)"
            )

        for offset in range(0, len(pcm), self.chunk_bytes):
            chunk = pcm[offset : offset + self.chunk_bytes]
            is_speech = self.classifier.is_speech(chunk)

            if not self.speech_active:
                if self._pre_roll_limit:
                    self._pre_roll.append(chunk)
                if not is_speech:
                    continue
                self.speech_active = True
                self._utterance = list(self._pre_roll) if self._pre_roll_limit else [chunk]
                self._speech_chunks = 1
                self._silence_chunks = 0
            else:
                self._utterance.append(chunk)
                if is_speech:
                    self._speech_chunks += 1
                    self._silence_chunks = 0
                else:
                    self._silence_chunks += 1

            if len(self._utterance) >= self._maximum_utterance_chunks:
                return self._finish_if_valid()

            if self._silence_chunks >= self._end_silence_chunks:
                return self._finish_if_valid()

        return None

    def _finish_if_valid(self) -> Optional[bytes]:
        utterance = b"".join(self._utterance)
        valid = self._speech_chunks >= self._minimum_speech_chunks
        self.reset()
        return utterance if valid else None

    def reset(self) -> None:
        self._pre_roll.clear()
        self._utterance = []
        self._speech_chunks = 0
        self._silence_chunks = 0
        self.speech_active = False
