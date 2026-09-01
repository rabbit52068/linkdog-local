"""Raw Opus codec helpers for LinkDog WebSocket v1 audio packets."""

from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path
from threading import Lock
from typing import Optional


OPUS_APPLICATION_AUDIO = 2049
OPUS_OK = 0
MAX_OPUS_PACKET_BYTES = 4_000


class OpusCodecError(RuntimeError):
    """Raised when libopus cannot encode or decode a LinkDog audio frame."""


def _load_libopus() -> ctypes.CDLL:
    candidates = [
        ctypes.util.find_library("opus"),
        "/opt/homebrew/opt/opus/lib/libopus.dylib",
        "/usr/local/opt/opus/lib/libopus.dylib",
        "/usr/lib/libopus.so.0",
        "/usr/lib/x86_64-linux-gnu/libopus.so.0",
    ]
    for candidate in candidates:
        if candidate and (not candidate.startswith("/") or Path(candidate).exists()):
            try:
                return ctypes.CDLL(candidate)
            except OSError:
                continue
    raise OpusCodecError(
        "libopus was not found; install the Opus library before starting voice audio"
    )


class OpusCodec:
    """Encode/decode fixed-duration mono PCM frames as raw Opus packets."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        frame_duration_ms: int = 60,
    ) -> None:
        if sample_rate <= 0 or channels not in (1, 2) or frame_duration_ms <= 0:
            raise ValueError("invalid Opus codec configuration")

        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_duration_ms = frame_duration_ms
        self.samples_per_channel = sample_rate * frame_duration_ms // 1000
        self.pcm_bytes_per_frame = self.samples_per_channel * channels * 2
        self._lib = _load_libopus()
        self._configure_functions()
        self._lock = Lock()
        self._encoder: Optional[int] = self._create_encoder()
        self._decoder: Optional[int] = self._create_decoder()

    def _configure_functions(self) -> None:
        self._lib.opus_encoder_create.argtypes = [
            ctypes.c_int32,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.opus_encoder_create.restype = ctypes.c_void_p
        self._lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
        self._lib.opus_encode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int32,
        ]
        self._lib.opus_encode.restype = ctypes.c_int32

        self._lib.opus_decoder_create.argtypes = [
            ctypes.c_int32,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.opus_decoder_create.restype = ctypes.c_void_p
        self._lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
        self._lib.opus_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.opus_decode.restype = ctypes.c_int

        self._lib.opus_strerror.argtypes = [ctypes.c_int]
        self._lib.opus_strerror.restype = ctypes.c_char_p

    def _error_text(self, code: int) -> str:
        return self._lib.opus_strerror(code).decode("utf-8", errors="replace")

    def _create_encoder(self) -> int:
        error = ctypes.c_int()
        encoder = self._lib.opus_encoder_create(
            self.sample_rate, self.channels, OPUS_APPLICATION_AUDIO, ctypes.byref(error)
        )
        if not encoder or error.value != OPUS_OK:
            raise OpusCodecError(f"failed to create Opus encoder: {self._error_text(error.value)}")
        return encoder

    def _create_decoder(self) -> int:
        error = ctypes.c_int()
        decoder = self._lib.opus_decoder_create(
            self.sample_rate, self.channels, ctypes.byref(error)
        )
        if not decoder or error.value != OPUS_OK:
            raise OpusCodecError(f"failed to create Opus decoder: {self._error_text(error.value)}")
        return decoder

    def encode(self, pcm: bytes) -> bytes:
        if len(pcm) != self.pcm_bytes_per_frame:
            raise OpusCodecError(
                f"PCM frame must contain exactly {self.samples_per_channel} samples "
                f"per channel ({self.pcm_bytes_per_frame} bytes)"
            )
        if self._encoder is None:
            raise OpusCodecError("Opus codec is closed")

        pcm_array = (ctypes.c_int16 * (len(pcm) // 2)).from_buffer_copy(pcm)
        output = (ctypes.c_ubyte * MAX_OPUS_PACKET_BYTES)()
        with self._lock:
            encoded_size = self._lib.opus_encode(
                self._encoder,
                pcm_array,
                self.samples_per_channel,
                output,
                MAX_OPUS_PACKET_BYTES,
            )
        if encoded_size < 0:
            raise OpusCodecError(f"Opus encode failed: {self._error_text(encoded_size)}")
        return bytes(output[:encoded_size])

    def decode(self, packet: bytes) -> bytes:
        if not packet:
            raise OpusCodecError("empty Opus packet")
        if self._decoder is None:
            raise OpusCodecError("Opus codec is closed")

        packet_array = (ctypes.c_ubyte * len(packet)).from_buffer_copy(packet)
        output_samples = self.samples_per_channel * self.channels
        output = (ctypes.c_int16 * output_samples)()
        with self._lock:
            decoded_samples = self._lib.opus_decode(
                self._decoder,
                packet_array,
                len(packet),
                output,
                self.samples_per_channel,
                0,
            )
        if decoded_samples < 0:
            raise OpusCodecError(f"Opus decode failed: {self._error_text(decoded_samples)}")
        if decoded_samples != self.samples_per_channel:
            raise OpusCodecError(
                f"unexpected decoded frame size: {decoded_samples} samples per channel"
            )
        return bytes(output)

    def close(self) -> None:
        with self._lock:
            if self._encoder is not None:
                self._lib.opus_encoder_destroy(self._encoder)
                self._encoder = None
            if self._decoder is not None:
                self._lib.opus_decoder_destroy(self._decoder)
                self._decoder = None

    def __enter__(self) -> "OpusCodec":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
