"""Action bridge: HTTP and voice requests to firmware MCP tool calls."""

import asyncio
import json
from typing import Any, Awaitable, Callable, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocketDisconnect
from pydantic import BaseModel

from app import state
from app.actions import ACTION_SPECS, build_arguments, result_type, tool_name
from app.auth import require_token
from app.device_session import DeviceSession, SessionClosedError
from app.voice_turn import VoiceActionError

router = APIRouter()


# Full action catalog mirroring the official repo (see app/actions.py).
# Keep ALLOWED_ACTIONS as a backward-compatible alias: action -> official MCP tool name.
ALLOWED_ACTIONS = {action: tool_name(action) for action in ACTION_SPECS}


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


def resolve_mcp_response(device_id: str, payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    response_id = payload.get("id")
    pending = state.PENDING_ACTIONS.get(response_id)
    if pending is None or pending[0] != device_id:
        return False
    response_future = pending[1]
    if response_future.done():
        return False
    response_future.set_result(payload)
    return True


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


@router.post("/xiaozhi/action", dependencies=[Depends(require_token)])
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
        session = state.ACTIVE_SESSIONS.get(device_id)
    elif len(state.ACTIVE_SESSIONS) == 1:
        device_id, session = next(iter(state.ACTIVE_SESSIONS.items()))
    else:
        device_id, session = None, None

    if session is None or device_id is None:
        raise HTTPException(status_code=409, detail="device is not connected")

    lock = state.ACTION_LOCKS.setdefault(device_id, asyncio.Lock())
    if lock.locked():
        raise HTTPException(status_code=409, detail="another action is already running")

    async with lock:
        # Motion actions need the C3 state gate. Read-only status and S3 hardware
        # volume are independent of servo state and should not pay this delay.
        if request.action not in {"get_device_status", "set_volume"}:
            await ensure_listening_state(session, device_id)

        request_id = next(state.REQUEST_IDS)
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
        state.PENDING_ACTIONS[request_id] = (device_id, response_future)
        try:
            await session.send_json(message)
            print(f"[ACTION] sent {request.action} to {device_id}, request_id={request_id}")
            payload = await asyncio.wait_for(
                asyncio.shield(response_future),
                timeout=state.ACTION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="device action timed out")
        except (RuntimeError, WebSocketDisconnect, SessionClosedError):
            if state.ACTIVE_SESSIONS.get(device_id) is session:
                state.ACTIVE_SESSIONS.pop(device_id, None)
            raise HTTPException(status_code=409, detail="device connection is closed")
        finally:
            state.PENDING_ACTIONS.pop(request_id, None)

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
    if not state.SETTINGS_STORE.path.exists():
        return
    settings = state.load_dashboard_settings()
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
