"""Single source of truth for TTS backend/voice configuration parsing.

This module centralises the environment parsing that was previously copied
across ``run_adapter.sh``, ``scripts/tts_preflight.py``, ``app/main.py`` and
``app/pocket_tts.py``. Every consumer must call :func:`resolve_backend`,
:func:`resolve_voice` and :func:`classify_voice` here instead of reading env
vars and re-implementing ``strip`` / ``lower`` / fallback logic on its own.

The module must stay import-light: it must NOT import ``pocket_tts`` (or any
heavy package) at module level, so that the preflight script and the health
snapshot can import it without triggering a model load.
"""

from __future__ import annotations

from urllib.parse import urlsplit

# Predefined Pocket TTS catalog voices. This mirrors the installed
# ``pocket_tts.utils.utils._ORIGINS_OF_PREDEFINED_VOICES`` keys (all lowercase).
# Any name not in this set is treated as a local file path / audio reference,
# so an unknown or misspelled voice fails loudly instead of silently falling
# back to a catalog voice.
CATALOG_VOICES: frozenset[str] = frozenset(
    {
        "cosette",
        "marius",
        "javert",
        "alba",
        "jean",
        "anna",
        "vera",
        "fantine",
        "charles",
        "paul",
        "eponine",
        "azelma",
        "george",
        "mary",
        "jane",
        "michael",
        "eve",
        "bill_boerst",
        "peter_yearsley",
        "stuart_bell",
        "caro_davy",
        "giovanni",
        "lola",
        "juergen",
        "rafael",
        "estelle",
    }
)

DEFAULT_CATALOG_VOICE = "cosette"


def resolve_backend(env: dict | None = None) -> str:
    """Return the normalised backend name: strip + lower; unset -> 'command'."""
    import os

    env = os.environ if env is None else env
    value = env.get("LINKDOG_TTS_BACKEND", "")
    normalized = (value or "").strip().lower()
    return normalized or "command"


def resolve_voice(env: dict | None = None) -> str:
    """Return the normalised voice.

    Empty values (unset, empty string, or pure whitespace) fall back to
    :data:`DEFAULT_CATALOG_VOICE`. A bare catalog name (no path separator, no
    extension) is canonicalised to its lowercase form so that both the runtime
    and the preflight hand the *same* string to ``get_state_for_audio_prompt``
    (whose catalog match is case-sensitive). Anything else is returned
    stripped, unchanged.
    """
    import os

    env = os.environ if env is None else env
    value = env.get("LINKDOG_POCKET_VOICE", "")
    stripped = (value or "").strip()
    if not stripped:
        return DEFAULT_CATALOG_VOICE
    has_separator = "/" in stripped or "\\" in stripped
    has_extension = "." in stripped
    if not has_separator and not has_extension:
        lowered = stripped.lower()
        if lowered in CATALOG_VOICES:
            return lowered
    return stripped


def _is_safetensors_source(source: str) -> bool:
    """Mirror upstream ``pocket_tts`` ``_is_safetensors_source`` semantics.

    A ``.safetensors`` reference is a serialised model state (usable directly
    via ``_import_model_state``) and must NOT be classified as voice cloning.
    """
    text = str(source)
    if text.startswith(("http://", "https://")):
        text = urlsplit(text).path
    elif text.startswith("hf://"):
        text = text.rsplit("@", 1)[0]
    return text.endswith(".safetensors")


# Sources that ``pocket_tts`` resolves itself (Hub download / HTTP fetch)
# rather than reading from the local filesystem.
_REMOTE_SCHEMES = ("hf://", "http://", "https://")


def is_remote_source(source: str) -> bool:
    """Return True when ``source`` is remote, not a local filesystem path.

    Callers that validate a voice must skip local ``Path.exists()`` checks for
    these: an ``hf://`` reference is downloaded at load time and is *never*
    present locally, so a local existence check would falsely reject a valid
    configuration (and the preflight aborts the adapter when it fails).
    """
    return str(source).startswith(_REMOTE_SCHEMES)


def classify_voice(voice: str) -> str:
    """Classify an already-normalised voice as 'catalog' | 'state' | 'audio'.

    ``voice`` must already have been passed through :func:`resolve_voice`.

    - Exact match in :data:`CATALOG_VOICES` -> 'catalog'.
    - A ``.safetensors`` source (local path, ``hf://``, or ``http(s)://``) ->
      'state' (serialised state, no voice cloning required).
    - Any other non-empty value -> 'audio' (requires voice cloning).
    - Empty -> 'audio' (callers should not pass empty; treated conservatively).
    """
    if voice in CATALOG_VOICES:
        return "catalog"
    if voice and _is_safetensors_source(voice):
        return "state"
    return "audio"


def voice_requires_cloning(voice: str) -> bool:
    """Return True when ``voice`` (already normalised) needs voice cloning."""
    return classify_voice(voice) == "audio"
