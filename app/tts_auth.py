"""Classify *why* the gated Pocket TTS weights are unreachable.

Motivation (root cause of the 2026-09-13 silent TTS failure)
------------------------------------------------------------
``pocket_tts.models.tts_model`` downloads the voice-cloning weights and, on
*failure*, silently degrades::

    try:
        weights_file = download_if_necessary(config.weights_path)
    except Exception:
        tts_model.has_voice_cloning = False
        weights_file = download_if_necessary(config.weights_path_without_voice_cloning)

(``tts_model.py`` lines ~168-172 and ~229-235.) The original exception is
discarded. The failure only resurfaces much later, from
``get_state_for_audio_prompt``, as the generic
``VOICE_CLONING_UNSUPPORTED`` message — which does not distinguish a missing
token, an expired token, unaccepted licence terms, or a network outage.

This module recovers that lost cause *before* the model load, so the operator
gets one precise sentence instead of a silent degradation. It is a diagnostic
only: it never downloads weights, never mutates model state, and never decides
availability by itself. ``has_voice_cloning`` on the loaded model remains the
authority.

Secrets
-------
The HuggingFace token is read via :func:`huggingface_hub.get_token` and passed
through ``hf_hub``/``httpx`` in memory. It is **never** logged, returned, or
interpolated into a message; every string this module produces is passed
through :func:`app.redact.redact_secrets` before it leaves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.redact import redact_secrets

# Gated repo holds the voice-cloning weights; the ungated sibling is what
# pocket_tts silently falls back to when the gated download fails.
GATED_REPO = "kyutai/pocket-tts"
UNGATED_REPO = "kyutai/pocket-tts-without-voice-cloning"

# Outcome vocabulary. Kept as plain strings so the values travel unchanged
# through logs, exit codes and human-readable summaries.
OK = "ok"
NO_TOKEN = "no_token"
INVALID_TOKEN = "invalid_token"
NO_ACCESS = "no_access"
NETWORK = "network"
UNKNOWN = "unknown"

# One sentence an operator can act on immediately, per outcome.
_REMEDY = {
    OK: "gated weights are reachable; voice cloning is available.",
    NO_TOKEN: (
        "no HuggingFace token was found. Run `hf auth login` (or place a token "
        "at ~/.cache/huggingface/token) so the gated weights can be downloaded."
    ),
    INVALID_TOKEN: (
        "the HuggingFace token was rejected (HTTP 401). It is expired or "
        "revoked; re-run `hf auth login`."
    ),
    NO_ACCESS: (
        "the token is valid but is not authorised for "
        f"{GATED_REPO} (HTTP 403/404). Accept the terms at "
        f"https://huggingface.co/{GATED_REPO} with the same account, then retry."
    ),
    NETWORK: (
        "could not reach huggingface.co. Check connectivity/DNS; if the weights "
        "are already cached this is harmless, otherwise the model load will fail."
    ),
    UNKNOWN: "could not determine gated-weight access; treating as unknown.",
}


@dataclass(frozen=True)
class GatedAccessDiagnosis:
    """Result of probing gated-weight access. Never carries secret material."""

    outcome: str
    detail: str = ""
    # Where the credential came from (app.hf_token provenance vocabulary).
    # Empty when the caller did not supply it. Never affects ``outcome``: a
    # cache-sourced token that works is still ``ok``, but reporting the source
    # keeps a masked durability regression visible.
    token_source: str = ""
    # Whether that source survives a cache clear. R6 (Astra round 2): reporting
    # the source alone was not enough — a consumer would have to know which
    # source names imply fragility. ``None`` means "not determined", which must
    # never be rendered as healthy.
    durable: Optional[bool] = None

    @property
    def ok(self) -> bool:
        return self.outcome == OK

    @property
    def token_present(self) -> bool:
        return self.outcome not in (NO_TOKEN,)

    def describe(self) -> str:
        """One-line, secret-free explanation suitable for stderr / logs."""
        remedy = _REMEDY.get(self.outcome, _REMEDY[UNKNOWN])
        parts = [f"{self.outcome}: {remedy}"]
        if self.detail:
            parts.append(f"(detail: {self.detail})")
        if self.token_source:
            parts.append(f"[token source: {self.token_source}]")
        return " ".join(parts)


def _with_source(
    diagnosis: GatedAccessDiagnosis,
    token_source: str | None,
    durable: Optional[bool] = None,
) -> GatedAccessDiagnosis:
    """Attach credential provenance to a diagnosis without altering it."""
    if not token_source or diagnosis.token_source:
        return diagnosis
    return GatedAccessDiagnosis(
        diagnosis.outcome,
        diagnosis.detail,
        token_source=token_source,
        durable=durable,
    )


def _classify_exception(exc: BaseException) -> GatedAccessDiagnosis:
    """Map a huggingface_hub/httpx exception onto an outcome.

    Imported lazily so this module stays importable when huggingface_hub or
    httpx are unavailable (e.g. a stripped test environment).
    """
    name = type(exc).__name__
    status = getattr(getattr(exc, "response", None), "status_code", None)

    # huggingface_hub raises these for token/authorisation problems. Their
    # names are stable; matching on them avoids a hard import dependency.
    if name == "GatedRepoError":
        return GatedAccessDiagnosis(NO_ACCESS, "GatedRepoError")
    if name == "RepositoryNotFoundError":
        # HF returns 404 for a gated/private repo when the token lacks access,
        # so this is an authorisation problem, not a typo in the repo id.
        return GatedAccessDiagnosis(NO_ACCESS, "RepositoryNotFoundError")
    if status == 401:
        return GatedAccessDiagnosis(INVALID_TOKEN, "HTTP 401")
    if status in (403, 404):
        return GatedAccessDiagnosis(NO_ACCESS, f"HTTP {status}")

    # Transport-level failures.
    if name in ("ConnectError", "ConnectTimeout", "ReadTimeout", "TimeoutException"):
        return GatedAccessDiagnosis(NETWORK, name)

    text = redact_secrets(str(exc)).lower()
    if "connection" in text or "timed out" in text or "name resolution" in text:
        return GatedAccessDiagnosis(NETWORK, name)
    if "unauthorized" in text:
        return GatedAccessDiagnosis(INVALID_TOKEN, name)

    return GatedAccessDiagnosis(UNKNOWN, f"{name}: {redact_secrets(str(exc))[:200]}")


def diagnose_gated_access(
    token: str | None = None,
    probes: dict | None = None,
    token_source: str | None = None,
) -> GatedAccessDiagnosis:
    """Probe whether the gated voice-cloning weights are reachable.

    Parameters
    ----------
    token:
        Token to use. When ``None`` it is read from the Hub in memory
        (``huggingface_hub.get_token``). The value is never logged.
    probes:
        Optional test seam. Recognised keys:

        - ``get_token``: ``() -> str | None``
        - ``whoami``: ``(token) -> None`` — raises on a rejected token
        - ``auth_check``: ``(repo_id, token) -> None`` — raises when the token
          is not authorised for the repo

        Any key omitted falls back to the real ``huggingface_hub`` call.
    token_source:
        Optional credential provenance (see ``app.hf_token``). Recorded on the
        diagnosis so a *cache-sourced* token is never mistaken for the durable
        configuration. Never affects the outcome — only the explanation.

    Returns
    -------
    GatedAccessDiagnosis
        ``OUTCOME`` plus a secret-free explanation. ``UNKNOWN`` is returned
        when the probe itself cannot run — callers must NOT treat an
        inconclusive probe as a hard failure.

    Why ``auth_check`` and not ``model_info``
    -----------------------------------------
    ``model_info`` returns HTTP 200 for the gated repo **without any token at
    all**: unauthenticated metadata is public. Using it as the authorisation
    probe produced a false ``ok`` on a machine with no usable credential
    (measured 2026-09-14: unauthenticated metadata 200, ``auth-check`` 401).
    ``GET /api/models/{repo}/auth-check`` is the endpoint that actually returns
    401/403 when the caller is not authorised, so it is the only probe whose
    success means what this function claims it means.
    """
    probes = probes or {}

    provenance_durable: Optional[bool] = None
    if token_source is None:
        # Auto-detect rather than default to "unknown": a caller that forgets
        # to pass provenance must not be able to hide a cache-sourced credential
        # behind an ``ok``. app.hf_token imports the Hub lazily, so this is safe.
        try:
            from app.hf_token import credential_provenance

            _provenance = credential_provenance()
            token_source = _provenance.source
            # R6: durability must travel WITH the source. Reporting only the
            # source pushed the "which source names are fragile" knowledge onto
            # every consumer, which is how it got lost in the first place.
            provenance_durable = _provenance.durable
        except Exception:  # noqa: BLE001 - provenance is explanatory only
            token_source = None

    get_token = probes.get("get_token")
    whoami = probes.get("whoami")
    auth_check = probes.get("auth_check")

    if get_token is None:
        try:
            from huggingface_hub import get_token as _hf_get_token

            get_token = _hf_get_token
        except Exception as exc:  # noqa: BLE001 - missing dep is inconclusive
            return GatedAccessDiagnosis(
                UNKNOWN, f"huggingface_hub unavailable: {type(exc).__name__}"
            )

    # Resolve the token in memory only. The value never leaves this function.
    if token is None:
        try:
            token = get_token()
        except Exception as exc:  # noqa: BLE001
            return _with_source(_classify_exception(exc), token_source, provenance_durable)

    if not token:
        # No token: the gated download *will* fail. This is the single most
        # common cause of the silent degradation.
        return _with_source(GatedAccessDiagnosis(NO_TOKEN), token_source, provenance_durable)

    if whoami is None or auth_check is None:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            if whoami is None:
                whoami = lambda tok: api.whoami(token=tok)  # noqa: E731
            if auth_check is None:

                def auth_check(repo, tok):  # noqa: E306 - test seam signature
                    api.auth_check(repo, token=tok)

        except Exception as exc:  # noqa: BLE001 - missing dep is inconclusive
            return _with_source(
                GatedAccessDiagnosis(
                    UNKNOWN, f"huggingface_hub unavailable: {type(exc).__name__}"
                ),
                token_source,
                provenance_durable,
            )

    # 1. Is the token itself accepted at all?
    try:
        whoami(token)
    except Exception as exc:  # noqa: BLE001
        return _with_source(_classify_exception(exc), token_source, provenance_durable)

    # 2. Is this token authorised for the gated repo specifically?
    #    auth-check is the endpoint that 401/403s an unauthorised caller;
    #    metadata reads succeed anonymously and prove nothing.
    try:
        auth_check(GATED_REPO, token)
    except Exception as exc:  # noqa: BLE001
        return _with_source(_classify_exception(exc), token_source, provenance_durable)

    return _with_source(GatedAccessDiagnosis(OK), token_source, provenance_durable)
