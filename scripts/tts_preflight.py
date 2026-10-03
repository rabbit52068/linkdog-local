#!/usr/bin/env python
"""Pre-flight check for the Pocket TTS path.

Runs before ``uvicorn`` starts (invoked unconditionally from ``run_adapter.sh``)
so that a silently-degraded TTS backend cannot ship into production: when the
gated weights fail to download, ``pocket_tts`` swallows the error and flips
``has_voice_cloning`` to ``False``, which breaks local-wav voice cloning while
``/health`` keeps reporting ok and nothing is logged.

This script is fail-closed only for the cases that actually break:
  - a local *audio* voice (wav/mp3/flac) that needs voice cloning but whose
    cloning is unavailable, or whose reference file is missing / unreadable;
  - a ``.safetensors`` state voice that cannot be imported.

Catalog voices do not need cloning and merely warn. Non-pocket backends are
skipped entirely (B4) *before* any Pocket classification or model access, so a
stale ``LINKDOG_POCKET_VOICE`` under ``edge``/``command`` can never reach the
model path.

All backend/voice parsing is delegated to :mod:`app.tts_config`, the single
source of truth shared with the runtime — so a voice that passes here is the
exact string the runtime will hand to ``get_state_for_audio_prompt``.

Two *independent* results
-------------------------
``readiness`` (the exit code: can TTS actually run?) and ``credential
durability`` (would it still run after a cache clear?) are reported separately,
because a ready preflight is frequently cache-backed — the weights are already
on disk, so no credential is read at all. The old script returned 0 and said
nothing about credentials, which implied a green light validated token
durability when it did not.

Every line is passed through :func:`app.redact.redact_secrets`, since these
messages interpolate a configured voice path and raw exception text, and Hub
errors routinely embed a request URL that can carry ``?token=...``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

# Make the project's ``app`` package importable when run as ``scripts/...``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.redact import redact_secrets  # noqa: E402
from app.tts_auth import diagnose_gated_access  # noqa: E402
from app.tts_config import (  # noqa: E402
    classify_voice,
    is_remote_source,
    resolve_backend,
    resolve_voice,
)


def load_model() -> Any:
    from pocket_tts import TTSModel

    return TTSModel.load_model()


def synthesize_short(model: Any, state: dict, text: str = "Linkdog is ready.") -> Any:
    """Synthesize one short clip and return the concatenated audio array."""
    import numpy as np

    chunks = []
    for chunk in model.generate_audio_stream(state, text, copy_state=True):
        if hasattr(chunk, "detach"):
            chunk = chunk.detach().cpu().float().numpy()
        chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
    if not chunks:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(chunks)


def _is_finite_and_not_silent(audio: Any) -> bool:
    """Reject all-NaN/Inf and all-silent audio (non-blocking hardening)."""
    import numpy as np

    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return False
    if not np.isfinite(arr).all():
        return False
    return bool(np.any(arr != 0.0))


def _emit(out: Any, text: str) -> None:
    """Write one line to ``out`` (a file-like object or a list of lines).

    Every line is passed through :func:`redact_secrets` first. Preflight output
    lands in the adapter log and in operator terminals, and the messages
    assembled below interpolate a configured voice path and raw exception text;
    Hub errors routinely embed a request URL that can carry ``?token=...``.
    Redacting centrally here covers every branch rather than each call site.
    """
    safe = redact_secrets(text)
    if isinstance(out, list):
        out.append(safe)
    else:
        print(safe, file=out)


def _credential_provenance() -> tuple[str, str]:
    """Return ``(durable?, detail)`` for the credential the Hub will actually use.

    Deliberately separate from the readiness result. Preflight can succeed
    *because the weights are already in the local cache*, which needs no
    credential at all — so a passing preflight is not evidence that the
    project-pinned token works, and must not be reported as if it were
    (second-review finding: the old script implied it validated token
    durability while returning 0 with no token present).

    Never returns secret material — only a provenance label.
    """
    try:
        from app.hf_token import credential_provenance

        prov = credential_provenance()
        return ("durable" if prov.durable else "not_durable"), prov.describe()
    except Exception as exc:  # noqa: BLE001 - reporting must never raise
        return "unknown", f"provenance probe failed ({type(exc).__name__})"


def run(
    model_factory: Callable[[], Any] = load_model,
    env: Optional[dict] = None,
    out: Any = None,
    auth_probe: Optional[Callable[[], Any]] = None,
) -> int:
    """Run the preflight. Returns the process exit code (0 = start, non-0 = abort).

    ``auth_probe`` is a test seam: it replaces :func:`diagnose_gated_access` so
    tests can exercise the gated-weight failure branch without touching the
    network or reading a real token.
    """
    env = os.environ if env is None else env
    if out is None:
        out = sys.stderr
    if auth_probe is None:
        auth_probe = diagnose_gated_access

    # B4: non-pocket backends skip at the very top, before any classification
    # or model access. A stale local voice under edge/command is ignored here.
    backend = resolve_backend(env)
    if backend != "pocket":
        _emit(out, f"[tts_preflight] backend={backend!r}: non-pocket, skipped")
        return 0

    voice = resolve_voice(env)
    voice_kind = classify_voice(voice)

    # B2/B5: catalog voices need no cloning and no model.
    if voice_kind == "catalog":
        _emit(
            out,
            f"[tts_preflight] WARNING: voice {voice!r} is a catalog voice "
            f"(no voice cloning required). Starting.",
        )
        return 0

    # A reference that must be resolved to state or audio. A non-empty voice
    # that is neither catalog nor state is treated as an audio reference.
    #
    # Remote sources (hf://, http(s)://) are resolved by pocket_tts at load
    # time and are never present on the local filesystem, so a local
    # Path.exists() check would falsely reject a valid configuration here —
    # and this script aborting means the adapter never starts. Only local
    # paths are checked for existence.
    remote_source = is_remote_source(voice)
    path_exists = True if remote_source else Path(voice).exists()

    if voice_kind == "state":
        # .safetensors: serialised state, imported directly, no cloning needed.
        if not path_exists:
            _emit(
                out,
                f"[tts_preflight] FATAL: configured state voice {voice!r} "
                f"does not exist. Fix LINKDOG_POCKET_VOICE to an existing "
                f".safetensors path, or a catalog voice name (e.g. cosette).",
            )
            return 1
    else:  # "audio" — requires voice cloning
        if not path_exists:
            _emit(
                out,
                f"[tts_preflight] FATAL: configured audio voice {voice!r} "
                f"does not exist. Fix LINKDOG_POCKET_VOICE to an existing "
                f"wav/mp3 path, or a catalog voice name (e.g. cosette).",
            )
            return 1

    try:
        model = model_factory()
    except Exception as exc:  # noqa: BLE001 - preflight must report, not raise
        _emit(out, f"[tts_preflight] FATAL: could not load Pocket TTS model: {exc}")
        return 1

    # .safetensors does not require voice cloning; only audio does (B5).
    if voice_kind == "audio":
        has_voice_cloning = bool(getattr(model, "has_voice_cloning", False))
        if not has_voice_cloning:
            # pocket_tts swallowed the real download error and silently fell
            # back to the ungated weights, so `has_voice_cloning` is False with
            # no explanation anywhere. Recover the cause ourselves: without it
            # the operator sees only VOICE_CLONING_UNSUPPORTED and has to guess.
            _emit(
                out,
                f"[tts_preflight] FATAL: local audio voice {voice!r} requires "
                f"voice cloning, which is unavailable "
                f"(model.has_voice_cloning=False).",
            )
            _emit(
                out,
                "[tts_preflight] pocket_tts silently fell back to the UNGATED "
                "weights after the gated download failed. Probing the reason...",
            )
            try:
                diagnosis = auth_probe()
                diagnosis_line = diagnosis.describe()
            except Exception as exc:  # noqa: BLE001 - diagnosis must never raise
                diagnosis_line = f"unknown: diagnosis itself failed ({type(exc).__name__})"
            _emit(out, f"[tts_preflight] gated-weight access: {diagnosis_line}")
            durability, durability_detail = _credential_provenance()
            _emit(
                out,
                f"[tts_preflight] credential: {durability} — {durability_detail}",
            )
            return 1

    # Exercise the real failure point: get_state_for_audio_prompt(...).
    try:
        state = model.get_state_for_audio_prompt(voice)
    except Exception as exc:  # noqa: BLE001
        kind = "state" if voice_kind == "state" else "audio"
        _emit(
            out,
            f"[tts_preflight] FATAL: get_state_for_audio_prompt({voice!r}) "
            f"({kind}) failed: {exc}",
        )
        return 1

    try:
        audio = synthesize_short(model, state)
        if not _is_finite_and_not_silent(audio):
            raise RuntimeError("synthesis produced empty/NaN/silent audio")
    except Exception as exc:  # noqa: BLE001
        _emit(
            out,
            f"[tts_preflight] FATAL: test synthesis produced no audio for "
            f"{voice!r}: {exc}",
        )
        return 1

    _emit(
        out,
        f"[tts_preflight] OK: voice {voice!r} ({voice_kind}) ready, "
        f"test synthesis produced {audio.size} samples.",
    )
    # Readiness and credential durability are different questions. A ready
    # preflight can be entirely cache-backed (weights already on disk need no
    # token), so report provenance separately instead of letting a green result
    # imply the credential was exercised.
    durability, durability_detail = _credential_provenance()
    if durability == "durable":
        _emit(out, f"[tts_preflight] credential: durable — {durability_detail}")
    else:
        _emit(
            out,
            f"[tts_preflight] credential: {durability} — {durability_detail} "
            f"(readiness above does not imply the credential was used)",
        )
    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
