"""Resident Pocket TTS backend for low-latency LinkDog speech."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Optional

import numpy as np
from scipy.signal import resample_poly

from app.redact import redact_secrets
from app.tts import TTSError
from app.tts_auth import diagnose_gated_access
from app.tts_config import classify_voice, voice_requires_cloning


def is_catalog_voice(voice: str) -> bool:
    """Return True when ``voice`` is a built-in name rather than a file path.

    Thin wrapper around :func:`app.tts_config.classify_voice` kept for
    backward compatibility; the catalog set lives only in ``tts_config``.
    """
    if not voice:
        return False
    return classify_voice(voice) == "catalog"


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
        auth_probe: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.voice = voice
        self._model_factory = model_factory or _default_model_factory
        self._auth_probe = auth_probe or diagnose_gated_access
        self._model: Optional[Any] = None
        self._state: Optional[Any] = None
        self._lock = threading.Lock()
        self.load_status = "idle"
        self.last_error: Optional[str] = None
        # Set only when a failed load looks like the gated-weight silent
        # degradation. Holds an outcome string (e.g. 'no_token'), never a secret.
        self.cloning_diagnosis: Optional[str] = None
        self.cloning_diagnosis_detail: Optional[str] = None
        # Where that credential came from, so a cache-sourced token reporting
        # 'ok' is distinguishable from a durable one.
        self.cloning_credential_source: Optional[str] = None
        # Whether that source survives a cache clear. ``None`` means "not
        # observed yet" — never "fine".
        self.cloning_credential_durable: Optional[bool] = None

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

    def warm(self) -> None:
        """Load the model and voice state now; failures surface via load_status."""
        with self._lock:
            self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load_status = "loading"
            try:
                model = self._model_factory()
                state = model.get_state_for_audio_prompt(self.voice)
            except Exception as exc:  # noqa: BLE001 - surface via load_status/last_error
                self.load_status = "failed"
                # Redacted: this string is published verbatim by
                # /api/health and the dashboard, and Hub
                # errors embed the request URL (which can carry ?token=...).
                self.last_error = redact_secrets(str(exc))

                # The load failed for a voice that needs cloning. pocket_tts
                # swallows the gated-download error and falls back to the
                # ungated weights, so the message above may be the generic
                # VOICE_CLONING_UNSUPPORTED with no cause. Recover the reason
                # once, here, where we know the failure actually happened —
                # never from /health, which must stay load-free and fast.
                if voice_requires_cloning(self.voice):
                    self.cloning_diagnosis, self.cloning_diagnosis_detail = (
                        self._diagnose_cloning()
                    )
                raise
            # Publish both only after state succeeds, so /health can never
            # observe a half-loaded backend as healthy.
            self._model = model
            self._state = state
            self.load_status = "ready"
            self.last_error = None

    def _diagnose_cloning(self) -> tuple[Optional[str], Optional[str]]:
        """Return ``(outcome, detail)`` for a failed cloning voice load.

        Never raises: a diagnostic that breaks the failure path is worse than
        no diagnostic at all. Returns ``(None, None)`` when inconclusive.

        Side effect: stores the credential source and durability on the
        instance so they reach ``/api/health``.
        """
        try:
            diagnosis = self._auth_probe()
        except Exception as exc:  # noqa: BLE001
            return None, redact_secrets(f"diagnosis failed: {type(exc).__name__}")
        outcome = getattr(diagnosis, "outcome", None)
        detail = getattr(diagnosis, "detail", None)
        # Source and durability describe ONE observation, so they are recorded
        # together. Round-3 review: writing them independently let a second
        # diagnosis (source=hub_cache, durable undetermined) leave the earlier
        # observation's ``durable=True`` in place — a stale verdict published as
        # if it were current. An explicit ``durable=False`` from M3 is still
        # recorded; only a *newer* observation with no verdict clears it to
        # ``None`` ("not observed"), never inheriting the older one.
        source = getattr(diagnosis, "token_source", None)
        durable = getattr(diagnosis, "durable", None)
        if source or durable is not None:
            self.cloning_credential_source = str(source) if source else None
            self.cloning_credential_durable = (
                bool(durable) if durable is not None else None
            )
        if not outcome:
            return None, None
        return outcome, redact_secrets(detail) if detail else None

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
