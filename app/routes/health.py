"""Public liveness probe and the token-protected TTS diagnostics."""

import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends

from app import state
from app.auth import require_token
from app.hf_token import credential_provenance
from app.redact import redact_secrets
from app.tts_config import classify_voice, resolve_backend, resolve_voice, voice_requires_cloning
from app.voice_turn import tts_failure_last, tts_failure_process, tts_failure_total

router = APIRouter()


@router.get("/health")
async def health():
    """Public liveness probe; diagnostics live behind /api/health."""
    return {
        "status": "ok",
        "connected_devices": sorted(state.ACTIVE_SESSIONS),
    }


@router.get("/api/health", dependencies=[Depends(require_token)])
async def api_health():
    return {
        "status": "ok",
        "connected_devices": sorted(state.ACTIVE_SESSIONS),
        "tts": _tts_health_snapshot(),
    }


def _credential_provenance_snapshot() -> Optional[Dict[str, Any]]:
    """Report where the Hub credential actually comes from, secret-free.

    Without this, ``configured_ok: true`` could coexist with "the credential
    is one cache clear from vanishing" and nothing would say so.

    Never raises: a broken probe must not take down /health.
    """
    try:
        provenance = credential_provenance()
    except Exception as exc:  # noqa: BLE001
        return {"source": "unknown", "durable": None, "degraded": None, "detail": f"provenance probe failed: {type(exc).__name__}"}
    return {
        "source": provenance.source,
        "durable": provenance.durable,
        "degraded": provenance.degraded,
        # Redacted as well: the detail embeds filesystem paths, and future
        # writers may extend it.
        "detail": redact_secrets(provenance.detail) if provenance.detail else "",
    }


def credential_source_from_backend(backend_obj: Any) -> Optional[str]:
    """The credential source the TTS backend actually observed, if any.

    ``None`` when nothing has been probed yet (no failed load), so this never
    fabricates a verdict. Kept separate from the live snapshot above: this is
    what the *backend* saw, which is the value a failure diagnosis used.
    """
    if backend_obj is None:
        return None
    source = getattr(backend_obj, "cloning_credential_source", None)
    return str(source) if source else None


def credential_durable_from_backend(backend_obj: Any) -> Optional[bool]:
    """The credential durability the TTS backend actually observed, if any.

    Companion to :func:`credential_source_from_backend`: a source alone can
    read as healthy while being one cache clear from failing, so the
    durability verdict travels alongside it.

    ``None`` when nothing has been probed yet (no failed load), so this never
    fabricates a verdict.
    """
    if backend_obj is None:
        return None
    durable = getattr(backend_obj, "cloning_credential_durable", None)
    return bool(durable) if durable is not None else None


def _tts_health_snapshot() -> Dict[str, Any]:
    """Report the TTS backend's real state without triggering a model load.

    Reading the resident backend is cheap; we only *inspect* the already-loaded
    model/state (if any) and never call ``load_model`` here, so /health stays
    fast and cannot be turned into an accidental model-loading endpoint.

    Three-stage semantics:
      - Nothing observed yet -> ``voice_cloning_available=None``,
        ``configured_ok=None``, ``model_status='unknown'`` (never ``false``).
      - ``ready`` requires BOTH model and state to be present; a model that
        failed its state prompt is ``failed``, not ``loaded``.
      - ``catalog`` voices can report ``configured_ok=True`` without a model,
        but a ``failed`` load is checked FIRST so a broken model can never be
        masked by the catalog branch.
      - ``last_error`` is passed through :func:`app.redact.redact_secrets` on
        the way out; this payload is rendered in browsers and copied into logs.
    """
    backend = resolve_backend()
    if backend == "pocket":
        voice = resolve_voice()
        voice_kind = classify_voice(voice)
        cloning_required = voice_requires_cloning(voice)

        backend_obj = state.POCKET_TTS_BACKEND
        load_status = getattr(backend_obj, "load_status", None) if (
            backend_obj is not None
        ) else None
        model = getattr(backend_obj, "_model", None) if backend_obj is not None else None
        voice_state = getattr(backend_obj, "_state", None) if backend_obj is not None else None
        last_error = getattr(backend_obj, "last_error", None) if backend_obj is not None else None

        if load_status == "ready" and model is not None and voice_state is not None:
            model_status = "ready"
            cloning_available = bool(getattr(model, "has_voice_cloning", False))
        elif load_status == "loading":
            model_status = "loading"
            cloning_available = None
        elif load_status == "failed":
            model_status = "failed"
            cloning_available = None
        else:
            # idle / unknown / no backend object yet: nothing observed.
            model_status = "unknown"
            cloning_available = None

        # Only ever populated on a failed load (see PocketTTSBackend); it
        # explains *why* cloning is unavailable instead of leaving an operator
        # with the generic VOICE_CLONING_UNSUPPORTED message.
        cloning_diagnosis = (
            getattr(backend_obj, "cloning_diagnosis", None)
            if backend_obj is not None
            else None
        )
        cloning_diagnosis_detail = (
            getattr(backend_obj, "cloning_diagnosis_detail", None)
            if backend_obj is not None
            else None
        )

        if model_status == "failed":
            # A failed model load is a real failure for EVERY voice kind,
            # catalog included. "Needs no cloning" != "model loads"; letting
            # the catalog branch answer first would reinstate exactly the
            # false-ok that this snapshot exists to eliminate.
            configured_ok = False
        elif voice_kind == "catalog":
            # Catalog voices need no cloning and no model to be known-good.
            configured_ok = True
        elif model_status in ("unknown", "loading"):
            # Nothing observed yet, or a load in flight: unknown, not a verdict.
            configured_ok = None
        else:  # ready
            configured_ok = (not cloning_required) or bool(cloning_available)

        return {
            "backend": "pocket",
            # The voice is operator-supplied and may be a URL with a credential
            # in its query string (…/voice.wav?token=...). Redact on the way out.
            "voice": redact_secrets(voice),
            "voice_kind": voice_kind,
            "voice_cloning_required": cloning_required,
            "voice_cloning_available": cloning_available,
            "configured_ok": configured_ok,
            "model_status": model_status,
            # Explains why voice cloning is unavailable (no_token / invalid_token
            # / no_access / network), instead of leaving only the generic
            # VOICE_CLONING_UNSUPPORTED text in last_error. Both are secret-free.
            "cloning_diagnosis": (
                redact_secrets(cloning_diagnosis) if cloning_diagnosis else None
            ),
            "cloning_diagnosis_detail": (
                redact_secrets(cloning_diagnosis_detail)
                if cloning_diagnosis_detail
                else None
            ),
            # Live view: where the credential comes from right now.
            "credential": _credential_provenance_snapshot(),
            # Observed view: what the backend saw when a load failed (None
            # until then). Source and durability travel together.
            "credential_source": credential_source_from_backend(backend_obj),
            "credential_durable": credential_durable_from_backend(backend_obj),
            # Defence in depth: the backend already redacts at the source, but
            # this payload reaches browsers and logs, so mask again
            # on the way out rather than trusting every future writer.
            "last_error": redact_secrets(last_error) if last_error else None,
            "tts_failures": tts_failure_total(),
            "tts_failure_last": tts_failure_last(),
            "tts_failure_process": tts_failure_process(),
        }

    return {
        "backend": backend,
        "voice": redact_secrets(os.environ.get("LINKDOG_TTS_VOICE", "")),
        "voice_kind": "n/a",
        "voice_cloning_required": False,
        "voice_cloning_available": None,
        "configured_ok": True,
        "model_status": "n/a",
        "cloning_diagnosis": None,
        "cloning_diagnosis_detail": None,
        "credential": _credential_provenance_snapshot(),
        "credential_source": None,
        "last_error": None,
        "tts_failures": tts_failure_total(),
        "tts_failure_last": tts_failure_last(),
        "tts_failure_process": tts_failure_process(),
    }
