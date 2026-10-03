import asyncio
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.actions import ACTION_SPECS, build_arguments, result_type, tool_name
from app.asr import FasterWhisperASR
from app.audio_codec import OpusCodec
from app.dashboard_settings import DEFAULT_MODEL, DashboardSettings, SettingsStore
from app.device_session import DeviceSession, SessionClosedError
from app.hermes_client import DEFAULT_SYSTEM_PROMPT, HermesAPIClient
from app.hf_token import credential_provenance
from app.model_catalog import CatalogUpstreamError, OllamaModelCatalog
from app.playback import OpusDownlinkPlayer
from app.pocket_tts import PocketTTSBackend
from app.tts import CommandTTSBackend
from app.redact import redact_secrets
from app.tts_config import classify_voice, resolve_backend, resolve_voice, voice_requires_cloning
from app.vad import UtteranceEndpoint, WebRtcVadClassifier
from app.voice_input import VoiceInputPipeline
from app.voice_turn import (
    VoiceActionError,
    VoiceTurnWorker,
    tts_failure_last,
    tts_failure_process,
    tts_failure_total,
)

app = FastAPI(title="hermes-linkdog")

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
DASHBOARD_DIR = Path(__file__).with_name("dashboard")
SETTINGS_STORE = SettingsStore(
    Path(os.environ.get(
        "LINKDOG_SETTINGS_PATH",
        str(Path(__file__).resolve().parent.parent / "data" / "settings.json"),
    ))
)
app.mount(
    "/dashboard/assets",
    StaticFiles(directory=str(DASHBOARD_DIR)),
    name="dashboard-assets",
)

# Full action catalog mirroring the official repo (see app/actions.py).
# Keep ALLOWED_ACTIONS as a backward-compatible alias: action -> official MCP tool name.
ALLOWED_ACTIONS = {action: tool_name(action) for action in ACTION_SPECS}
ACTIVE_SESSIONS: Dict[str, DeviceSession] = {}
REQUEST_IDS = itertools.count(1)
ACTION_TIMEOUT_SECONDS = float(os.environ.get("LINKDOG_ACTION_TIMEOUT", "8"))
PENDING_ACTIONS: Dict[int, Tuple[str, asyncio.Future]] = {}
ACTION_LOCKS: Dict[str, asyncio.Lock] = {}
_POCKET_TTS_BACKEND: Any = None
MODEL_CATALOG = OllamaModelCatalog(os.environ.get("LINKDOG_HERMES_API_KEY", ""))

VOICE_ACTIONS = ("sit_down", "stand_up", "get_down", "shake_hands")
VOICE_ACTION_CONFIRMATIONS = {
    "sit_down": "Okay, I sat down.",
    "stand_up": "Okay, I'm standing up.",
    "get_down": "Okay, I'm lying down.",
    "shake_hands": "Here, shake!",
}
VOICE_ACTION_TOOL = [{
    "type": "function",
    "function": {
        "name": "linkdog_action",
        "description": (
            "Only call this when the user asks the robot dog to perform an action "
            "right now. Do not call for negations, questions about capability, or "
            "descriptions of past or future actions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(VOICE_ACTIONS),
                }
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "linkdog_volume",
        "description": (
            "Control the robot dog's hardware speaker volume. Use mode 'up' for "
            "requests such as louder, raise the volume, turn it up, or increase "
            "the volume. Use mode 'down' for quieter, lower the volume, turn it "
            "down, or decrease the volume. Use mode 'set' with an exact volume "
            "from 10 to 100. Use minimum or maximum for those explicit requests."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["set", "up", "down", "minimum", "maximum"],
                },
                "volume": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 100,
                },
            },
            "required": ["mode"],
            "additionalProperties": False,
        },
    },
}]


class ActionRequest(BaseModel):
    action: str
    device_id: Optional[str] = None
    # Parameters passed through by action type (duration / times / speed / part+angle / mode / gesture)
    duration: Optional[int] = None
    times: Optional[int] = None
    speed: Optional[int] = None
    volume: Optional[int] = None
    part: Optional[str] = None
    angle: Optional[int] = None
    mode: Optional[int] = None
    gesture: Optional[int] = None
    name: Optional[str] = None


class DashboardSettingsRequest(BaseModel):
    agent_name: str
    system_prompt: str
    model: str
    api_url: str = ""
    memory_enabled: bool
    max_history_turns: int
    user_profile: str = ""
    context_memory: str = ""
    volume: int


def load_dashboard_settings() -> DashboardSettings:
    """Load saved settings, falling back to the existing environment config."""
    if SETTINGS_STORE.path.exists():
        return SETTINGS_STORE.load()
    return DashboardSettings(
        agent_name="Xiaobin",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        model=os.environ.get("LINKDOG_HERMES_MODEL", DEFAULT_MODEL),
        api_url=os.environ.get("LINKDOG_HERMES_API_URL", ""),
        memory_enabled=True,
        max_history_turns=int(
            os.environ.get("LINKDOG_HERMES_HISTORY_TURNS", "6")
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


def resolve_mcp_response(device_id: str, payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    response_id = payload.get("id")
    pending = PENDING_ACTIONS.get(response_id)
    if pending is None or pending[0] != device_id:
        return False
    response_future = pending[1]
    if response_future.done():
        return False
    response_future.set_result(payload)
    return True


def build_voice_input(
    session: DeviceSession,
    idle_timeout_seconds: Optional[float] = None,
    on_idle_timeout: Optional[Callable[[], None]] = None,
) -> VoiceInputPipeline:
    if idle_timeout_seconds is None:
        idle_timeout_seconds = float(
            os.environ.get("LINKDOG_IDLE_TIMEOUT_SECONDS", "60")
        )
    codec = OpusCodec(sample_rate=16_000, channels=1, frame_duration_ms=60)
    endpoint = UtteranceEndpoint(
        classifier=WebRtcVadClassifier(sample_rate=16_000, aggressiveness=3),
        sample_rate=16_000,
        chunk_duration_ms=20,
        pre_roll_ms=300,
        minimum_speech_ms=300,
        end_silence_ms=440,
        maximum_utterance_ms=12_000,
    )
    return VoiceInputPipeline(
        session=session,
        codec=codec,
        endpoint=endpoint,
        idle_timeout_seconds=idle_timeout_seconds,
        on_idle_timeout=on_idle_timeout,
    )


async def disconnect_device(session: DeviceSession) -> None:
    """Send a WebSocket close frame, then release all per-device resources."""
    try:
        await session.websocket.close(code=1000)
    finally:
        await session.close()


def build_asr() -> FasterWhisperASR:
    language = _resolve_asr_language(os.environ.get("LINKDOG_ASR_LANGUAGE", "zh"))
    return FasterWhisperASR(
        model_name=os.environ.get("LINKDOG_ASR_MODEL", "base"),
        device=os.environ.get("LINKDOG_ASR_DEVICE", "cpu"),
        compute_type=os.environ.get("LINKDOG_ASR_COMPUTE_TYPE", "int8"),
        language=language,
        timeout_seconds=float(os.environ.get("LINKDOG_ASR_TIMEOUT", "15")),
        initial_prompt=os.environ.get("LINKDOG_ASR_INITIAL_PROMPT") or None,
    )


def _resolve_asr_language(raw: Optional[str]) -> Optional[str]:
    """Map a configured language to a valid faster-whisper language code.

    faster-whisper has no "auto" code; omitting the argument is what triggers
    language detection. Passing the literal string "auto" raises ValueError on
    every transcription, so empty and "auto" both become None here.
    """
    value = (raw or "").strip()
    if not value or value.lower() == "auto":
        return None
    return value


def build_tts() -> Any:
    global _POCKET_TTS_BACKEND
    backend = resolve_backend()
    if backend == "pocket":
        voice = resolve_voice()
        if _POCKET_TTS_BACKEND is None or _POCKET_TTS_BACKEND.voice != voice:
            _POCKET_TTS_BACKEND = PocketTTSBackend(voice=voice)
        return _POCKET_TTS_BACKEND

    return build_command_tts()


def build_command_tts() -> CommandTTSBackend:
    default_edge_tts = str(Path(sys.executable).with_name("edge-tts"))
    return CommandTTSBackend(
        voice=os.environ.get("LINKDOG_TTS_VOICE", "en-US-AriaNeural"),
        fallback_voice=os.environ.get("LINKDOG_TTS_FALLBACK_VOICE", "Samantha"),
        edge_tts_command=os.environ.get("LINKDOG_EDGE_TTS_COMMAND", default_edge_tts),
        ffmpeg_command=os.environ.get("LINKDOG_FFMPEG_COMMAND", "ffmpeg"),
        say_command=os.environ.get("LINKDOG_SAY_COMMAND", "say"),
    )


def build_player(session: DeviceSession) -> OpusDownlinkPlayer:
    codec = OpusCodec(sample_rate=16_000, channels=1, frame_duration_ms=60)
    return OpusDownlinkPlayer(session, codec, frame_duration_ms=60)


def build_hermes_client() -> HermesAPIClient:
    settings = load_dashboard_settings()
    return HermesAPIClient(
        base_url=settings.api_url.strip() or os.environ.get(
            "LINKDOG_HERMES_API_URL",
            "http://127.0.0.1:8642/v1",
        ),
        api_key=os.environ.get("LINKDOG_HERMES_API_KEY", ""),
        model=settings.model,
        provider=os.environ.get("LINKDOG_HERMES_PROVIDER", "ollama-cloud") or None,
        system_prompt=build_system_prompt(settings),
        max_history_turns=(
            settings.max_history_turns if settings.memory_enabled else 0
        ),
        timeout_seconds=float(os.environ.get("LINKDOG_HERMES_TIMEOUT", "60")),
        tools=VOICE_ACTION_TOOL,
        allowed_tool_actions=set(VOICE_ACTIONS),
    )


def voice_input_enabled() -> bool:
    return os.environ.get("LINKDOG_VOICE_INPUT_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def handle_device_event(
    device_id: str,
    voice_input: Optional[VoiceInputPipeline],
    event: Dict[str, Any],
    voice_turn: Optional[VoiceTurnWorker] = None,
) -> Optional[asyncio.Task]:
    if (
        voice_input is not None
        and event.get("type") == "listen"
        and event.get("state") == "start"
    ):
        # Only start on the "not-listening → listening" transition; the device re-sends start,
        # and resetting every time would drop the accumulating speech, so VAD never triggers.
        if not voice_input.is_listening:
            voice_input.start_listening()
    elif event.get("type") == "abort" and voice_turn is not None:
        reason = str(event.get("reason") or "unknown")
        return voice_turn.session.start_task(voice_turn.abort(reason))
    elif event.get("type") == "mcp":
        resolve_mcp_response(device_id, event.get("payload"))
    return None


async def ensure_listening_state(session: DeviceSession, device_id: str) -> None:
    """送 action 前，先把設備切到 Listening（AI status = 2）。

    根因：設備停在 Idle（AI status = 1）時，C3 會自動 getDown() + 關 servo
    power。若此時直接送 get_down／wiggle_tail，會與 C3 的自動 transition 競爭，
    造成 S3 reset。Xiaozhi 的正確做法是先讓設備進 Listening（C3 停在 sitDown
    穩定狀態），再送 action。

    透過 tts:start → tts:stop 驅動設備 Idle → Speaking → Listening。
    """
    await session.send_json({"type": "tts", "state": "start"})
    await asyncio.sleep(0.5)
    await session.send_json({"type": "tts", "state": "stop"})
    # After tts:stop the device waits WaitForPlayCompletion(1000) before switching to Listening; leave extra buffer.
    await asyncio.sleep(2.5)
    print(f"[STATE] {device_id} set to Listening before action")


@app.post("/xiaozhi/action")
async def send_action(request: ActionRequest):
    mcp_tool = ALLOWED_ACTIONS.get(request.action)
    if mcp_tool is None:
        raise HTTPException(status_code=400, detail="action is not allow-listed")

    # Build official MCP arguments by action type (with parameter passthrough and clamping).
    params = {
        "duration": request.duration,
        "times": request.times,
        "speed": request.speed,
        "volume": request.volume,
        "part": request.part,
        "angle": request.angle,
        "mode": request.mode,
        "gesture": request.gesture,
        "name": request.name,
    }
    params = {k: v for k, v in params.items() if v is not None}
    try:
        arguments = build_arguments(request.action, **params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if request.device_id:
        device_id = request.device_id
        session = ACTIVE_SESSIONS.get(device_id)
    elif len(ACTIVE_SESSIONS) == 1:
        device_id, session = next(iter(ACTIVE_SESSIONS.items()))
    else:
        device_id, session = None, None

    if session is None or device_id is None:
        raise HTTPException(status_code=409, detail="device is not connected")

    lock = ACTION_LOCKS.setdefault(device_id, asyncio.Lock())
    if lock.locked():
        raise HTTPException(status_code=409, detail="another action is already running")

    async with lock:
        # Motion actions need the C3 state gate. Read-only status and S3 hardware
        # volume are independent of servo state and should not pay this delay.
        if request.action not in {"get_device_status", "set_volume"}:
            await ensure_listening_state(session, device_id)

        request_id = next(REQUEST_IDS)
        message = {
            "session_id": f"session-{device_id}",
            "type": "mcp",
            "payload": {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": mcp_tool,
                    "arguments": arguments,
                },
                "id": request_id,
            },
        }
        response_future = asyncio.get_running_loop().create_future()
        PENDING_ACTIONS[request_id] = (device_id, response_future)
        try:
            await session.send_json(message)
            print(f"[ACTION] sent {request.action} to {device_id}, request_id={request_id}")
            payload = await asyncio.wait_for(
                asyncio.shield(response_future),
                timeout=ACTION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="device action timed out")
        except (RuntimeError, WebSocketDisconnect, SessionClosedError):
            if ACTIVE_SESSIONS.get(device_id) is session:
                ACTIVE_SESSIONS.pop(device_id, None)
            raise HTTPException(status_code=409, detail="device connection is closed")
        finally:
            PENDING_ACTIONS.pop(request_id, None)

        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict) or result.get("isError") is not False:
            raise HTTPException(status_code=502, detail="device reported action failure")

        # Determine success by return type:
        #   action — success returns "true" (bool), failure returns "false" or an error string
        #   text   — success returns the string content (query tools)
        content_items = [
            item for item in result.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if not content_items:
            raise HTTPException(status_code=502, detail="device reported action failure")

        text_value = content_items[0].get("text")
        if result_type(request.action) == "action":
            if text_value != "true":
                raise HTTPException(status_code=502, detail="device reported action failure")
            response_text = None
        else:
            # text type: return the string content
            response_text = text_value

        print(f"[ACTION] completed {request.action} on {device_id}, request_id={request_id}")
        return {
            "status": "completed",
            "device_id": device_id,
            "action": request.action,
            "request_id": request_id,
            "text": response_text,
        }


async def execute_voice_action(device_id: str, action: str) -> str:
    """Execute one LLM-selected allow-listed action through the MCP bridge."""
    if action not in VOICE_ACTION_CONFIRMATIONS:
        raise HTTPException(status_code=400, detail="voice action is not allow-listed")
    await send_action(ActionRequest(device_id=device_id, action=action))
    return VOICE_ACTION_CONFIRMATIONS[action]


async def execute_voice_volume(device_id: str, arguments: Dict[str, Any]) -> str:
    """Apply original-firmware hardware volume semantics for voice commands."""
    mode = arguments.get("mode")
    if mode == "set":
        target = arguments.get("volume")
    elif mode == "minimum":
        target = 10
    elif mode == "maximum":
        target = 100
    elif mode in {"up", "down"}:
        status = await send_action(ActionRequest(
            device_id=device_id,
            action="get_device_status",
        ))
        try:
            status_data = json.loads(status["text"])
            current = status_data["audio_speaker"]["volume"]
            if isinstance(current, bool) or not isinstance(current, int):
                raise ValueError("invalid current volume")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=502,
                detail="device returned invalid volume status",
            ) from exc
        target = current + (10 if mode == "up" else -10)
    else:
        raise HTTPException(status_code=400, detail="invalid volume mode")

    if isinstance(target, bool) or not isinstance(target, int):
        raise HTTPException(status_code=400, detail="invalid volume value")
    target = max(10, min(100, target))
    await send_action(ActionRequest(
        device_id=device_id,
        action="set_volume",
        volume=target,
    ))
    return f"Volume set to {target} percent."


async def apply_saved_volume(device_id: str) -> None:
    """Apply the dashboard preference after a device establishes its session."""
    if not SETTINGS_STORE.path.exists():
        return
    settings = load_dashboard_settings()
    try:
        await execute_voice_volume(
            device_id,
            {"mode": "set", "volume": settings.volume},
        )
    except HTTPException as exc:
        print(
            f"[DASHBOARD] volume apply failed for {device_id}: {exc.detail}"
        )


def build_voice_action_executor(device_id: str) -> Callable[[str], Awaitable[str]]:
    async def execute(action: str) -> str:
        try:
            return await execute_voice_action(device_id, action)
        except HTTPException as exc:
            raise VoiceActionError(str(exc.detail)) from exc

    return execute


def build_voice_volume_executor(
    device_id: str,
) -> Callable[[Dict[str, Any]], Awaitable[str]]:
    async def execute(arguments: Dict[str, Any]) -> str:
        try:
            return await execute_voice_volume(device_id, arguments)
        except HTTPException as exc:
            raise VoiceActionError(str(exc.detail)) from exc

    return execute


@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    return FileResponse(DASHBOARD_DIR / "index.html", media_type="text/html")


@app.get("/api/models")
async def get_models():
    try:
        result = await MODEL_CATALOG.get_models()
    except CatalogUpstreamError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "models": sorted(result.models, key=str.casefold),
        "stale": result.stale,
    }


@app.get("/api/settings")
async def get_settings():
    try:
        settings = load_dashboard_settings()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "settings": settings.to_dict(),
        "connected_devices": sorted(ACTIVE_SESSIONS),
        "connected_device_details": [
            {
                "device_id": device_id,
                "ip_address": getattr(ACTIVE_SESSIONS[device_id], "ip_address", None),
            }
            for device_id in sorted(ACTIVE_SESSIONS)
        ],
    }


@app.put("/api/settings")
async def update_settings(request: DashboardSettingsRequest):
    try:
        settings = DashboardSettings.from_dict(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        catalog = await MODEL_CATALOG.get_models()
    except CatalogUpstreamError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if settings.model not in catalog.models:
        raise HTTPException(status_code=422, detail="model is not available")

    SETTINGS_STORE.save(settings)

    connected_devices = sorted(ACTIVE_SESSIONS)
    volume_results = await asyncio.gather(*(
        execute_voice_volume(
            device_id,
            {"mode": "set", "volume": settings.volume},
        )
        for device_id in connected_devices
    ), return_exceptions=True)
    volume_applied_to = [
        device_id
        for device_id, result in zip(connected_devices, volume_results)
        if not isinstance(result, Exception)
    ]
    volume_failed_for = [
        device_id
        for device_id, result in zip(connected_devices, volume_results)
        if isinstance(result, Exception)
    ]
    return {
        "settings": settings.to_dict(),
        "connected_devices": connected_devices,
        "connected_device_details": [
            {
                "device_id": device_id,
                "ip_address": getattr(ACTIVE_SESSIONS[device_id], "ip_address", None),
            }
            for device_id in connected_devices
        ],
        "restart_required": bool(connected_devices),
        "volume_applied_to": volume_applied_to,
        "volume_failed_for": volume_failed_for,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "connected_devices": sorted(ACTIVE_SESSIONS),
        "tts": _tts_health_snapshot(),
    }


def _credential_provenance_snapshot() -> Optional[Dict[str, Any]]:
    """Report where the Hub credential actually comes from, secret-free.

    R6 (Astra round 2): the durability fix is only observable if provenance
    reaches an operator. It used to die inside ``pocket_tts``, which kept just
    ``(outcome, detail)`` — so ``configured_ok: true`` could coexist with
    "the credential is one cache clear from vanishing" and nothing said so.

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

    Companion to :func:`credential_source_from_backend`. Round-2 review (R6)
    showed that a source alone is not enough: ``outcome='ok'`` with
    ``source='hub_cache'`` means the credential works today and is one cache
    clear from failing, so the durability verdict has to travel alongside it or
    the operator sees a healthy-looking row that is not durable.

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

    Three-stage semantics (Astra blocking #3):
      - Nothing observed yet -> ``voice_cloning_available=None``,
        ``configured_ok=None``, ``model_status='unknown'`` (never ``false``).
      - ``ready`` requires BOTH model and state to be present; a model that
        failed its state prompt is ``failed``, not ``loaded``.
      - ``catalog`` voices can report ``configured_ok=True`` without a model,
        but a ``failed`` load is checked FIRST so a broken model can never be
        masked by the catalog branch.
      - ``last_error`` is passed through :func:`app.redact.redact_secrets` on
        the way out; this endpoint is unauthenticated and dashboard-rendered.
    """
    backend = resolve_backend()
    if backend == "pocket":
        voice = resolve_voice()
        voice_kind = classify_voice(voice)
        cloning_required = voice_requires_cloning(voice)

        backend_obj = _POCKET_TTS_BACKEND
        load_status = getattr(backend_obj, "load_status", None) if (
            backend_obj is not None
        ) else None
        model = getattr(backend_obj, "_model", None) if backend_obj is not None else None
        state = getattr(backend_obj, "_state", None) if backend_obj is not None else None
        last_error = getattr(backend_obj, "last_error", None) if backend_obj is not None else None

        if load_status == "ready" and model is not None and state is not None:
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
            # R3 (Astra round 2): the voice string reaches /health verbatim. It is
            # operator-supplied and may be a full URL carrying a credential in the
            # query string (LINKDOG_POCKET_VOICE=https://host/voice.wav?token=...),
            # and /health is unauthenticated. Redact on the way out — this leak
            # needs no exception and would survive any fix confined to
            # pocket_tts/last_error.
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
            # R6 (Astra round 2): where the credential ACTUALLY comes from, end to
            # end. The backend used to keep only (outcome, detail), so provenance
            # died inside pocket_tts and "ok" could still mean "one cache clear
            # from failing". Both values are secret-free by construction.
            "credential": _credential_provenance_snapshot(),
            "credential_source": credential_source_from_backend(backend_obj),
            # R6: durability must travel with the source. A working-but-cache-
            # sourced credential reports outcome='ok'; without this the row
            # reads healthy while it is one cache clear from failing.
            "credential_durable": credential_durable_from_backend(backend_obj),
            # Defence in depth: the backend already redacts at the source, but
            # /health is unauthenticated and dashboard-rendered, so mask again
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


# Song list: return an empty list so the device does not fetch from linkdog.me
@app.get("/xiaozhi/music/list.json")
async def music_list():
    return JSONResponse({"version": 0, "songs": []})


# Custom 1.8.15 is deployed; return the same version to avoid repeated OTA (force=1 forces reflash/downgrade).
@app.get("/xiaozhi/ota/esp32s3/firmware.json")
async def s3_firmware():
    return JSONResponse({
        "latest": {
            "version": "1.8.15",
            "url": f"http://{HOST}:{PORT}/xiaozhi/ota/esp32s3/linkdog-s3-ota_1.8.15.bin",
        }
    })


# S3 custom firmware download (kept for later USB or other recovery flows)
@app.get("/xiaozhi/ota/esp32s3/linkdog-s3-ota_1.8.15.bin")
async def s3_firmware_bin():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "..", "firmware", "linkdog-s3-ota_1.8.15.bin"),
        media_type="application/octet-stream",
        filename="linkdog-s3-ota_1.8.15.bin",
    )


# Stock ota_0 app extracted from the repo's 16MB merged image (embedded version 1.8.12).
@app.get("/xiaozhi/ota/esp32s3/linkdog-s3-stock_1.8.12-ota.bin")
async def s3_stock_firmware_bin():
    return FileResponse(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "firmware",
            "linkdog-s3-stock_1.8.12-ota.bin",
        ),
        media_type="application/octet-stream",
        filename="linkdog-s3-stock_1.8.12-ota.bin",
    )


# C3 firmware manifest: return current version 2.0.3 (= device version, no C3 upgrade triggered)
@app.get("/xiaozhi/ota/esp32c3/firmware.json")
async def c3_firmware():
    return JSONResponse({
        "latest": {
            "version": "2.0.3",
            "url": f"http://{HOST}:{PORT}/xiaozhi/ota/esp32c3/linkdog-c3_2.0.3.bin",
        }
    })


@app.post("/xiaozhi/ota/")
async def ota_bootstrap(request: Request):
    # Device POSTs device JSON; body content is not parsed at this stage, only the device id is logged
    body = await request.body()
    device_id = request.headers.get("device-id", "unknown")
    print(f"[OTA] bootstrap from {device_id}, body={len(body)} bytes")

    # Critical: return only the websocket section, never mqtt (otherwise the device uses MQTT)
    return JSONResponse({
        "websocket": {
            "url": f"ws://{HOST}:{PORT}/xiaozhi/ws",
            "version": 1,
        }
    })


@app.websocket("/xiaozhi/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    # Read the device hello
    raw = await ws.receive_text()
    hello = json.loads(raw)
    device_id = ws.headers.get("device-id", "unknown")
    print(f"[WS] hello from {device_id}: type={hello.get('type')}, "
          f"audio={hello.get('audio_params')}")

    # Send server hello (transport must be websocket, otherwise the device rejects it)
    await ws.send_text(json.dumps({
        "type": "hello",
        "transport": "websocket",
        "session_id": f"session-{device_id}",
        "audio_params": {
            "format": "opus",
            "sample_rate": 16000,   # match device hardware to avoid resampling
            "channels": 1,
            "frame_duration": 60,
        },
    }))
    print(f"[WS] server hello sent, session={device_id}")
    previous_session = ACTIVE_SESSIONS.get(device_id)
    if previous_session is not None:
        await previous_session.close()
    client_ip = ws.client.host if ws.client is not None else None
    session = DeviceSession(
        device_id=device_id,
        websocket=ws,
        ip_address=client_ip,
    )
    voice_input = build_voice_input(session) if voice_input_enabled() else None
    voice_turn = None
    if voice_input is not None:
        player = build_player(session)
        session.add_close_callback(player.close)
        voice_turn = VoiceTurnWorker(
            session,
            voice_input,
            build_asr(),
            hermes=build_hermes_client(),
            tts=build_tts(),
            player=player,
            action_executor=build_voice_action_executor(device_id),
            volume_executor=build_voice_volume_executor(device_id),
            disconnect=lambda: disconnect_device(session),
            abort_cooldown_seconds=float(
                os.environ.get("LINKDOG_ABORT_COOLDOWN", "2.0")
            ),
        )
        voice_input.on_idle_timeout = lambda: session.start_task(
            voice_turn.handle_idle_timeout()
        )
        session.start_task(voice_input.run())
        session.start_task(voice_turn.run())
    ACTIVE_SESSIONS[device_id] = session
    if SETTINGS_STORE.path.exists():
        session.start_task(apply_saved_volume(device_id))

    # Keep the connection, handle control messages, and count audio frames
    audio_frame_count = 0

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.receive":
                text = msg.get("text")
                if text:
                    print(f"[WS] text: {text[:500]}")
                    try:
                        event = json.loads(text)
                    except json.JSONDecodeError:
                        continue

                    handle_device_event(
                        device_id,
                        voice_input,
                        event,
                        voice_turn=voice_turn,
                    )
                elif msg.get("bytes") is not None:
                    packet = msg["bytes"]
                    if voice_input is not None:
                        session.enqueue_audio(packet)
                    audio_frame_count += 1
                    if audio_frame_count == 1 or audio_frame_count % 100 == 0:
                        size = len(packet)
                        print(f"[WS] audio: frames={audio_frame_count}, latest={size} bytes")
            elif msg.get("type") == "websocket.disconnect":
                break
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if ACTIVE_SESSIONS.get(device_id) is session:
            ACTIVE_SESSIONS.pop(device_id, None)
        await session.close()
        for request_id, (pending_device_id, response_future) in list(PENDING_ACTIONS.items()):
            if pending_device_id == device_id and not response_future.done():
                response_future.set_exception(WebSocketDisconnect())
                PENDING_ACTIONS.pop(request_id, None)
    print(f"[WS] {device_id} disconnected, audio_frames={audio_frame_count}")
