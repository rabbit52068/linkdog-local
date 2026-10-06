"""Device WebSocket session and the per-connection voice pipeline wiring."""

import asyncio
import logging
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app import state
from app.asr import FasterWhisperASR
from app.audio_codec import OpusCodec
from app.chat_client import ChatClient
from app.device_session import DeviceSession
from app.playback import OpusDownlinkPlayer
from app.pocket_tts import PocketTTSBackend
from app.routes.control import (
    VOICE_ACTION_TOOL,
    VOICE_ACTIONS,
    apply_saved_volume,
    build_voice_action_executor,
    build_voice_volume_executor,
    resolve_mcp_response,
)
from app.tts import CommandTTSBackend
from app.tts_config import resolve_backend, resolve_voice
from app.vad import UtteranceEndpoint, WebRtcVadClassifier
from app.voice_input import VoiceInputPipeline
from app.voice_turn import VoiceTurnWorker

LOGGER = logging.getLogger(__name__)

router = APIRouter()


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
        # Background noise can hold the VAD open; cap the wait at 7 s.
        maximum_utterance_ms=7_000,
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
    """Return the process-wide ASR backend, rebuilding it only on config change.

    Building one per connection made every reconnect (the device drops the
    socket after each idle timeout) reload the model on its first utterance.
    """
    config = dict(
        model_name=os.environ.get("LINKDOG_ASR_MODEL", "base"),
        device=os.environ.get("LINKDOG_ASR_DEVICE", "cpu"),
        compute_type=os.environ.get("LINKDOG_ASR_COMPUTE_TYPE", "int8"),
        language=_resolve_asr_language(
            os.environ.get("LINKDOG_ASR_LANGUAGE", "auto")
        ),
        timeout_seconds=float(os.environ.get("LINKDOG_ASR_TIMEOUT", "15")),
        initial_prompt=os.environ.get("LINKDOG_ASR_INITIAL_PROMPT") or None,
    )
    cached = state.ASR_BACKEND
    if cached is None or state.ASR_CONFIG != config:
        state.ASR_BACKEND = FasterWhisperASR(**config)
        state.ASR_CONFIG = config
    return state.ASR_BACKEND


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
    backend = resolve_backend()
    if backend == "pocket":
        voice = resolve_voice()
        if state.POCKET_TTS_BACKEND is None or state.POCKET_TTS_BACKEND.voice != voice:
            state.POCKET_TTS_BACKEND = PocketTTSBackend(voice=voice)
        return state.POCKET_TTS_BACKEND

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


def build_chat_client() -> ChatClient:
    settings = state.load_dashboard_settings()
    return ChatClient(
        base_url=settings.api_url.strip()
        or state.chat_env("API_URL", "http://127.0.0.1:8642/v1"),
        api_key=state.chat_env("API_KEY", ""),
        model=settings.model,
        provider=state.chat_env("PROVIDER") or None,
        system_prompt=state.build_system_prompt(settings),
        max_history_turns=(
            settings.max_history_turns if settings.memory_enabled else 0
        ),
        timeout_seconds=float(state.chat_env("TIMEOUT", "60")),
        tools=VOICE_ACTION_TOOL,
        allowed_tool_actions=set(VOICE_ACTIONS),
        history_path=state.HISTORY_PATH,
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
    elif (
        voice_input is not None
        and event.get("type") == "listen"
        and event.get("state") == "detect"
    ):
        # A wake word was spoken: its tail must be dropped from the next
        # listening window. Auto-mode follow-ups have no wake word.
        voice_input.note_wake_word()
    elif event.get("type") == "abort" and voice_turn is not None:
        # The device aborts playback when it hears the wake word.
        if voice_input is not None:
            voice_input.note_wake_word()
        reason = str(event.get("reason") or "unknown")
        return voice_turn.session.start_task(voice_turn.abort(reason))
    elif event.get("type") == "mcp":
        resolve_mcp_response(device_id, event.get("payload"))
    return None


@router.websocket("/xiaozhi/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    # Read the device hello
    raw = await ws.receive_text()
    try:
        hello = json.loads(raw)
    except json.JSONDecodeError:
        hello = None
    if not isinstance(hello, dict):
        LOGGER.info("[WS] rejected connection: first message is not a JSON object")
        await ws.close(code=1003)
        return
    device_id = ws.headers.get("device-id", "unknown")
    LOGGER.info(f"[WS] hello from {device_id}: type={hello.get('type')}, "
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
    LOGGER.info(f"[WS] server hello sent, session={device_id}")
    previous_session = state.ACTIVE_SESSIONS.get(device_id)
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
            chat=build_chat_client(),
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
    state.ACTIVE_SESSIONS[device_id] = session
    if state.SETTINGS_STORE.path.exists():
        session.start_task(apply_saved_volume(device_id))

    # Keep the connection, handle control messages, and count audio frames
    audio_frame_count = 0

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.receive":
                text = msg.get("text")
                if text:
                    LOGGER.info(f"[WS] text: {text[:500]}")
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
                        LOGGER.info(f"[WS] audio: frames={audio_frame_count}, latest={size} bytes")
            elif msg.get("type") == "websocket.disconnect":
                break
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if state.ACTIVE_SESSIONS.get(device_id) is session:
            state.ACTIVE_SESSIONS.pop(device_id, None)
        await session.close()
        for request_id, (pending_device_id, response_future) in list(state.PENDING_ACTIONS.items()):
            if pending_device_id == device_id and not response_future.done():
                response_future.set_exception(WebSocketDisconnect())
                state.PENDING_ACTIONS.pop(request_id, None)
    LOGGER.info(f"[WS] {device_id} disconnected, audio_frames={audio_frame_count}")
