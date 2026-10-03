"""Pin the HuggingFace token to a project-local file, before the Hub reads it.

Why this module exists (the 2026-09-13 silent TTS failure)
----------------------------------------------------------
``huggingface_hub`` resolves its token location **once, at import time**:

    huggingface_hub/constants.py:  HF_TOKEN_PATH = os.getenv("HF_TOKEN_PATH", ...)

so setting ``HF_TOKEN_PATH`` in ``os.environ`` *after* the first
``import huggingface_hub`` has no effect. Verified directly::

    >>> import os; from huggingface_hub import constants
    >>> constants.HF_TOKEN_PATH = '/Users/.../.cache/huggingface/token'
    >>> os.environ['HF_TOKEN_PATH'] = '/somewhere/else'
    >>> constants.HF_TOKEN_PATH     # unchanged

That is why this must run as early as possible, and why doing it inside a
function that imports the Hub first is useless.

The failure mode it prevents
----------------------------
``pocket_tts`` downloads gated voice-cloning weights through the Hub. By
default the token lives at ``~/.cache/huggingface/token`` — *inside the cache
directory*. Any cache-clearing step (``rm -rf ~/.cache/huggingface``, a Hub
cache prune, a fresh ``hf auth logout``) takes the token with it. The gated
download then fails, ``pocket_tts`` swallows the error and silently degrades to
the ungated weights, and TTS dies with ``/health`` still reporting ``ok`` and
nothing in the log. That is exactly what happened on 2026-09-13.

Pointing ``HF_TOKEN_PATH`` at a file outside the cache directory decouples
credential lifetime from cache lifetime, so clearing the cache no longer
destroys the credential.

Why provenance is reported, not just success (second review, 2026-09-14)
-----------------------------------------------------------------------
Pinning the path is not enough on its own. If the project file goes missing
while a token still sits in the Hub cache, the Hub silently falls back to the
cache credential: the download works, ``diagnose_gated_access`` reports ``ok``,
and the durability fix is **already broken without anyone noticing**. The next
cache clear is then another outage.

So this module also answers *where the credential actually comes from*.
:func:`credential_provenance` compares the path the Hub froze at import
(``constants.HF_TOKEN_PATH``) against the project path and classifies it. It
never needs the token *value* to do this, only the path. A cache-sourced
credential is reported as ``degraded``: working now, one cache-clear away from
failing.

Precedence
----------
An explicit ``HF_TOKEN_PATH`` in the environment always wins — an operator or a
test harness that sets it deliberately is never overridden here. The default is
also skipped when it points at a missing or empty file, so a stripped checkout
degrades to the Hub's own behaviour (``no_token``, precisely reported by
``app.tts_auth``) instead of silently reading a blank credential.

Secrets
-------
This module moves the *path*, not the secret: the token is never exported into
the environment, never returned, and never logged. The single read of file
content exists only to answer "is this file empty?" — the bytes are discarded
immediately and never retained, compared, or returned.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Project-local credential location. The repo-root .gitignore excludes
# ``secrets/``, and a regression test asserts that stays true.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The project credential file. Patchable by tests (see TESTING below); the
# ``LINKDOG_HF_TOKEN_FILE`` env var is the out-of-process equivalent, so a test
# harness can supply a synthetic fixture instead of requiring a real
# credential to exist in the checkout.
DEFAULT_TOKEN_PATH = PROJECT_ROOT / "secrets" / "hf_token"

_TOKEN_FILE_ENV = "LINKDOG_HF_TOKEN_FILE"

# Set only for bookkeeping/observability; never required by the Hub.
_BOOKKEEPING_KEY = "LINKDOG_HF_TOKEN_PATH"

# Pin outcomes — *why* pin_hf_token_path did or did not apply. "Deferred to an
# explicit setting" and "the project file has vanished" are very different
# situations, and only the latter is a regression.
PINNED = "pinned"
EXPLICIT_ENV = "explicit_env"
MISSING_FILE = "missing_file"
BLANK_FILE = "blank_file"
UNREADABLE = "unreadable"

# Credential provenance — where the Hub will actually read the token from.
SOURCE_PROJECT = "project"
SOURCE_EXPLICIT = "explicit"
SOURCE_HUB_CACHE = "hub_cache"
SOURCE_OTHER = "other"
SOURCE_NONE = "none"
# ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` in the environment outranks the
# pinned file, so a token can be working while the durability fix is bypassed.
SOURCE_ENV_TOKEN = "env_token"

# Last outcome of pin_hf_token_path(), for observability and /health.
LAST_PIN_OUTCOME: str | None = None


@dataclass(frozen=True)
class CredentialProvenance:
    """Where the Hub credential comes from, and whether that is durable."""

    source: str
    durable: bool
    degraded: bool
    hub_token_path: str = ""
    detail: str = ""

    def describe(self) -> str:
        """One-line, secret-free explanation suitable for stderr / logs."""
        return self.detail or self.source


def token_path(env: dict | None = None) -> Path:
    """Resolve the project credential path, honouring the ops/test seam."""
    env = os.environ if env is None else env
    override = (env.get(_TOKEN_FILE_ENV) or "").strip()
    if override:
        return Path(override).expanduser()
    return DEFAULT_TOKEN_PATH


def _hub_constants():
    """Return the frozen ``huggingface_hub.constants`` module, or ``None``."""
    try:
        from huggingface_hub import constants
    except Exception:  # noqa: BLE001 - a missing dep is not an error here
        return None
    return constants


def _hub_cache_dir(constants) -> Path | None:
    """The Hub cache directory — the one whose deletion destroys the default token."""
    home = getattr(constants, "HF_HOME", None)
    if isinstance(home, str) and home:
        return Path(home)
    return None


def _read_token_file(path: Path) -> str | None:
    """Return the stripped token stored at ``path``, or ``None``.

    Used ONLY for equality comparison inside :func:`credential_provenance`; the
    value is never logged, returned to callers, or published. The public
    provenance dataclass carries a boolean verdict, never the bytes.
    """
    try:
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    return raw or None


def resolve_hub_token(env: dict | None = None) -> str:
    """Return the token the Hub would use for a request, or ``""``.

    This is the single impure step of provenance: the one place that consults
    the live Hub. Kept as its own function so tests can inject a synthetic
    resolution (there is no supported way to make ``huggingface_hub.get_token``
    return fixture data, and the real environment on a dev machine already has a
    working credential — which would make every provenance test pass for the
    wrong reason).

    The value is used only for comparison inside :func:`credential_provenance`
    and is never logged or returned to a caller.
    """
    try:
        import huggingface_hub
    except Exception:  # noqa: BLE001
        return ""
    try:
        token = huggingface_hub.get_token()
    except Exception:  # noqa: BLE001 - never let a probe break startup
        return ""
    return str(token).strip() if token else ""


def _same_file(a, b) -> bool:
    """True when two paths name the same real file (symlinks resolved).

    Used to decide whether the file the Hub actually reads *is* the project
    credential. Comparing path strings is not enough: a symlink at the project
    path and the Hub's real path can differ textually while naming one file.
    """
    try:
        return Path(a).resolve() == Path(b).resolve()
    except (OSError, ValueError):
        return False


def _classify_resolved_token(token: str, env: dict | None = None) -> str:
    """Name the source that produced ``token``; ``""`` when it is empty.

    Pure classification of an already-resolved value against the candidate
    sources, ordered by the Hub's real precedence
    (``huggingface_hub/utils/_auth.py``)::

        OIDC -> HF_TOKEN/HUGGING_FACE_HUB_TOKEN env -> file_refreshed -> Colab
             -> HF_TOKEN_PATH file

    The order matters: a token present in BOTH the environment and the project
    file is reported as ``env_token``, because that is the one the Hub used and
    the one whose durability is compromised.

    Round-3 review (2026-09-14) fixed a *same value, different origin* false
    positive: the project file's value used to be compared FIRST, so an
    identical token sitting in both the project file and the Hub cache was
    reported as ``project``/durable even when the Hub read the cache copy.
    The file the Hub will really read is now consulted first, and only then the
    project file as a fallback for when that path is unreadable.
    """
    token = (token or "").strip()
    if not token:
        return ""

    env = os.environ if env is None else env

    # The Hub reads ``HF_TOKEN`` first, then the legacy name — note the real
    # spelling is ``HUGGING_FACE_HUB_TOKEN`` (round-3 finding: we had
    # ``HUGGINGFACE_HUB_TOKEN``, which the Hub never looks at, so a token set
    # under the legacy variable was misreported as coming from the project).
    explicit_env = (
        env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN") or ""
    ).strip()
    if explicit_env and token == explicit_env:
        return SOURCE_ENV_TOKEN

    project_path = token_path(env)

    # The file the Hub will actually read at request time. This is the
    # authoritative answer: it decides the source even when the project file
    # holds the very same value.
    hub_path = getattr(_hub_constants(), "HF_TOKEN_PATH", None)
    if hub_path:
        try:
            hub_token = _read_token_file(Path(str(hub_path)))
        except (OSError, ValueError):
            hub_token = None
        if hub_token and token == hub_token:
            # Same value, but WHICH file is it? A symlink into the cache
            # directory reports as the Hub cache, because clearing the cache
            # would destroy it.
            if _is_in_cache(str(hub_path), _hub_constants()):
                return SOURCE_HUB_CACHE
            if _same_file(hub_path, project_path):
                return SOURCE_PROJECT
            explicit_path = (env.get("HF_TOKEN_PATH") or "").strip()
            if explicit_path and _same_file(explicit_path, hub_path):
                return SOURCE_EXPLICIT
            return SOURCE_OTHER

    # Fallback only when the Hub's own path could not be read (absent, blank,
    # or the constant is unset). The project file is then the best evidence.
    project_token = _read_token_file(project_path)
    if project_token and token == project_token:
        if _is_in_cache(str(project_path)):
            return SOURCE_HUB_CACHE
        return SOURCE_PROJECT

    # A token the Hub resolved from a source we cannot see (OIDC, Colab, a
    # refreshed file). Not ours, so not durable by our definition.
    return SOURCE_OTHER


def _resolved_token_source(env: dict | None = None) -> str:
    """Probe the Hub, then classify what it chose. ``""`` when no credential."""
    env = os.environ if env is None else env
    return _classify_resolved_token(resolve_hub_token(env), env)


def _is_in_cache(path_str: str, constants=None) -> bool:
    """True when ``path_str`` resolves inside the Hub cache directory."""
    if constants is None:
        constants = _hub_constants()
    cache_dir = _hub_cache_dir(constants) if constants is not None else None
    if cache_dir is None:
        return False
    try:
        return Path(path_str).resolve().is_relative_to(cache_dir.resolve())
    except (OSError, ValueError):
        return False


def credential_provenance(env: dict | None = None) -> CredentialProvenance:
    """Classify where the Hub credential actually resolves from.

    The whole point of the durability fix is that the credential survives a
    cache clear. A token that still resolves out of the Hub cache directory is
    therefore reported as ``degraded`` even though it currently works — that is
    exactly the masking failure this check exists to surface.

    Round-2 review (2026-09-14) replaced the original *path-string comparison*
    with an actual resolution probe. Path comparison produced three false
    positives, all reproduced against the real Hub:

      1. Project file deleted, ``constants.HF_TOKEN_PATH`` still frozen at the
         project path -> reported ``project``/``durable`` while ``get_token()``
         returned ``None`` (no credential at all).
      2. A symlink at the project path pointing into the Hub cache -> reported
         ``durable`` although clearing the cache destroys it.
      3. ``HF_TOKEN`` set in the environment -> the Hub used it, the report still
         said ``project``, and the durability fix was bypassed.

    The fix asks the Hub which token it will actually use, then matches that
    value against the candidate sources. Only a boolean verdict leaves this
    module; the token value is used for comparison and then discarded.
    """
    env = os.environ if env is None else env
    constants = _hub_constants()
    if constants is None:
        return CredentialProvenance(
            source=SOURCE_NONE,
            durable=False,
            degraded=False,
            detail="huggingface_hub unavailable; credential provenance unknown",
        )

    hub_path = getattr(constants, "HF_TOKEN_PATH", None)
    hub_path_str = str(hub_path) if hub_path else ""

    resolved = _resolved_token_source(env)

    if not resolved:
        # Nothing will be read at request time. This is the honest answer for a
        # stripped checkout AND for the deleted-file case that used to claim
        # ``project``/``durable``.
        return CredentialProvenance(
            source=SOURCE_NONE,
            durable=False,
            degraded=False,
            hub_token_path=hub_path_str,
            detail=(
                "the Hub resolves no credential; gated downloads will need "
                "authentication. "
                + (
                    f"Expected the project file at {token_path(env)}."
                    if not hub_path_str
                    else f"The Hub token path is {hub_path_str}."
                )
            ),
        )

    project_str = str(token_path(env))

    if resolved == SOURCE_PROJECT:
        return CredentialProvenance(
            source=SOURCE_PROJECT,
            durable=True,
            degraded=False,
            hub_token_path=hub_path_str,
            detail=(
                "credential comes from the project file; it survives a cache "
                "clear (this is the durable configuration)."
            ),
        )

    if resolved == SOURCE_ENV_TOKEN:
        # Works right now, but the durability fix is bypassed: the credential
        # lives in the process environment, so a cache clear is not the only
        # thing that can remove it.
        return CredentialProvenance(
            source=SOURCE_ENV_TOKEN,
            durable=False,
            degraded=True,
            hub_token_path=hub_path_str,
            detail=(
                "DEGRADED: the Hub is using HF_TOKEN from the environment, which "
                "outranks the pinned project file — the durability fix is NOT in "
                "effect. Unset it to fall back to the project credential."
            ),
        )

    if resolved == SOURCE_HUB_CACHE:
        project_present = token_path(env).is_file()
        return CredentialProvenance(
            source=SOURCE_HUB_CACHE,
            durable=False,
            degraded=True,
            hub_token_path=hub_path_str,
            detail=(
                "DEGRADED: the Hub is reading the credential from inside its "
                "cache directory, so the durability fix is NOT in effect — a "
                "cache clear will break gated downloads again"
                + (
                    ". The project credential file exists but is not the one the "
                    "Hub selected; check import order (app/__init__.py must run "
                    "before any huggingface_hub import) and that no HF_TOKEN env "
                    "var is set."
                    if project_present
                    else f". Create {project_str} to restore durability."
                )
            ),
        )

    if resolved == SOURCE_EXPLICIT:
        in_cache = _is_in_cache(hub_path_str, constants)
        return CredentialProvenance(
            source=SOURCE_EXPLICIT,
            durable=not in_cache,
            degraded=in_cache,
            hub_token_path=hub_path_str,
            detail=(
                "credential path was set explicitly via HF_TOKEN_PATH"
                + (
                    ", but it points inside the Hub cache directory, so a cache "
                    "clear will still destroy the credential."
                    if in_cache
                    else " (operator-chosen, treated as durable)."
                )
            ),
        )

    # A token the Hub resolved from a source we cannot classify (OIDC, Colab,
    # a refreshed credential file). Durability is NOT ours to assert.
    #
    # Round-3 finding: this used to publish ``durable = not in_cache``, where
    # ``in_cache`` was computed from an ``HF_TOKEN_PATH`` that may have had
    # nothing to do with the value the Hub actually resolved. "Durability cannot
    # be verified here" must not be published as a confident ``durable=True`` —
    # an unverifiable source is reported as not durable, which is the honest
    # direction to fail in (it can only cause a warning, never mask a break).
    in_cache = _is_in_cache(hub_path_str, constants) if hub_path_str else False
    return CredentialProvenance(
        source=SOURCE_OTHER,
        durable=False,
        degraded=in_cache,
        hub_token_path=hub_path_str,
        detail=(
            "credential comes from a source outside the project file "
            f"({resolved}); durability CANNOT be verified, so it is not "
            "reported as durable."
            + (
                " It resolves inside the Hub cache directory, so a cache clear "
                "will destroy it."
                if in_cache
                else ""
            )
        ),
    )


def pin_hf_token_path(env: dict | None = None) -> bool:
    """Point ``HF_TOKEN_PATH`` at the project-local token file.

    Must be called **before** ``huggingface_hub`` is first imported.

    Returns ``True`` when this call chose the project-local path, ``False``
    when it deferred to an existing ``HF_TOKEN_PATH`` or to the file being
    absent/empty. Never raises: a credential-location problem must not stop the
    adapter from starting — ``app.tts_auth`` reports the real outcome instead.

    The reason for a ``False`` return is recorded in :data:`LAST_PIN_OUTCOME`,
    because "deferred" and "the project file vanished" are very different
    situations and only the latter is a regression.
    """
    global LAST_PIN_OUTCOME  # noqa: PLW0603

    env = os.environ if env is None else env

    # An explicit choice by an operator or test harness always wins.
    if env.get("HF_TOKEN_PATH"):
        LAST_PIN_OUTCOME = EXPLICIT_ENV
        return False

    path = token_path(env)

    try:
        if not path.is_file():
            LAST_PIN_OUTCOME = MISSING_FILE
            return False
        if not path.read_bytes().strip():
            # Present but blank: leave the Hub to report no_token honestly
            # rather than pinning an empty credential.
            LAST_PIN_OUTCOME = BLANK_FILE
            return False
    except OSError:
        LAST_PIN_OUTCOME = UNREADABLE
        return False

    env["HF_TOKEN_PATH"] = str(path)
    env[_BOOKKEEPING_KEY] = str(path)
    LAST_PIN_OUTCOME = PINNED
    return True
