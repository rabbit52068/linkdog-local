"""Process-wide adapter state shared by the route modules.

Route modules access these as ``state.NAME`` (never ``from app.state import``)
so that tests can patch a single attribute and every caller sees it.
"""

import asyncio
import itertools
import os
from pathlib import Path
from typing import Any, Dict, Tuple

from app.chat_client import DEFAULT_SYSTEM_PROMPT
from app.dashboard_settings import DEFAULT_MODEL, DashboardSettings, SettingsStore
from app.device_session import DeviceSession
from app.model_catalog import OllamaModelCatalog


def chat_env(name: str, default: str = "") -> str:
    """Read LINKDOG_CHAT_<name>, falling back to the legacy LINKDOG_HERMES_<name>.

    The voice path talks to an OpenAI-compatible endpoint directly, not to
    Hermes; the old names are still honoured so existing .env files keep working.
    """
    for key in (f"LINKDOG_CHAT_{name}", f"LINKDOG_HERMES_{name}"):
        value = os.environ.get(key)
        if value is not None:
            return value
    return default


def _detect_lan_ip() -> str:
    """Return this host's LAN IP by opening a UDP socket to a public address."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


# Host LAN IP used in firmware URLs. Auto-detected unless LINKDOG_HOST is set.
HOST = os.environ.get("LINKDOG_HOST") or _detect_lan_ip()


PORT = os.environ.get("LINKDOG_PORT", "8003")


SETTINGS_STORE = SettingsStore(
    Path(os.environ.get(
        "LINKDOG_SETTINGS_PATH",
        str(Path(__file__).resolve().parent.parent / "data" / "settings.json"),
    ))
)


HISTORY_PATH = Path(os.environ.get(
    "LINKDOG_HISTORY_PATH",
    str(Path(__file__).resolve().parent.parent / "data" / "history.json"),
))


ACTIVE_SESSIONS: Dict[str, DeviceSession] = {}


REQUEST_IDS = itertools.count(1)


ACTION_TIMEOUT_SECONDS = float(os.environ.get("LINKDOG_ACTION_TIMEOUT", "8"))


PENDING_ACTIONS: Dict[int, Tuple[str, asyncio.Future]] = {}


ACTION_LOCKS: Dict[str, asyncio.Lock] = {}


# Resident Pocket TTS backend, shared by every session and read by /api/health.
POCKET_TTS_BACKEND: Any = None


MODEL_CATALOG = OllamaModelCatalog(chat_env("API_KEY", ""))


def load_dashboard_settings() -> DashboardSettings:
    """Load saved settings, falling back to the existing environment config."""
    if SETTINGS_STORE.path.exists():
        return SETTINGS_STORE.load()
    return DashboardSettings(
        agent_name="Xiaobin",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model=chat_env("MODEL", DEFAULT_MODEL),
        api_url=chat_env("API_URL", ""),
        memory_enabled=True,
        max_history_turns=int(
            chat_env("HISTORY_TURNS", "6")
        ),
        volume=int(os.environ.get("LINKDOG_DEFAULT_VOLUME", "70")),
    )


def build_system_prompt(settings: DashboardSettings) -> str:
    sections = [settings.system_prompt.strip()]
    if settings.memory_enabled and settings.user_profile.strip():
        sections.append("User profile:\n" + settings.user_profile.strip())
    if settings.memory_enabled and settings.context_memory.strip():
        sections.append("Context memory:\n" + settings.context_memory.strip())
    return "\n\n".join(section for section in sections if section)
