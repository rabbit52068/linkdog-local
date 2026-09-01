import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import app.main as main
from app.device_session import DeviceSession
from app.dashboard_settings import SettingsStore
from app.main import (
    build_asr,
    build_hermes_client,
    build_player,
    build_tts,
    build_voice_action_executor,
    build_voice_volume_executor,
    build_voice_input,
    disconnect_device,
    handle_device_event,
    voice_input_enabled,
)
from app.asr import FasterWhisperASR
from app.hermes_client import HermesAPIClient
from app.playback import OpusDownlinkPlayer
from app.pocket_tts import PocketTTSBackend
from app.tts import CommandTTSBackend
from app.voice_input import VoiceInputPipeline


class FakeWebSocket:
    def __init__(self):
        self.close_calls = []

    async def send_text(self, _text):
        pass

    async def send_bytes(self, _payload):
        pass

    async def close(self, code=1000):
        self.close_calls.append(code)


class VoiceWiringTests(unittest.TestCase):
    def test_voice_input_is_disabled_by_default(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(voice_input_enabled())

    def test_voice_input_requires_explicit_true_value(self):
        with patch.dict("os.environ", {"LINKDOG_VOICE_INPUT_ENABLED": "true"}):
            self.assertTrue(voice_input_enabled())

    def test_builds_default_16khz_voice_input_pipeline(self):
        session = DeviceSession("TEST:DOG", FakeWebSocket())

        pipeline = build_voice_input(session)

        self.assertIsInstance(pipeline, VoiceInputPipeline)
        self.assertEqual(pipeline.codec.sample_rate, 16_000)
        self.assertEqual(pipeline.codec.frame_duration_ms, 60)
        self.assertEqual(pipeline.endpoint.chunk_duration_ms, 20)
        self.assertEqual(pipeline.endpoint._end_silence_chunks, 22)
        pipeline.codec.close()

    def test_builds_idle_timeout_from_environment(self):
        session = DeviceSession("TEST:DOG", FakeWebSocket())

        with patch.dict(
            "os.environ", {"LINKDOG_IDLE_TIMEOUT_SECONDS": "12.5"}, clear=True
        ):
            pipeline = build_voice_input(session)

        self.assertEqual(pipeline.idle_timeout_seconds, 12.5)
        pipeline.codec.close()

    def test_disconnect_closes_websocket_and_session(self):
        async def scenario():
            websocket = FakeWebSocket()
            session = DeviceSession("TEST:DOG", websocket)

            await disconnect_device(session)

            self.assertEqual(websocket.close_calls, [1000])
            self.assertTrue(session.closed)

        asyncio.run(scenario())

    def test_builds_asr_from_environment(self):
        values = {
            "LINKDOG_ASR_MODEL": "small",
            "LINKDOG_ASR_DEVICE": "cpu",
            "LINKDOG_ASR_COMPUTE_TYPE": "int8",
            "LINKDOG_ASR_LANGUAGE": "zh",
            "LINKDOG_ASR_TIMEOUT": "9.5",
        }
        with patch.dict("os.environ", values, clear=True):
            backend = build_asr()

        self.assertIsInstance(backend, FasterWhisperASR)
        self.assertEqual(backend.model_name, "small")
        self.assertEqual(backend.device, "cpu")
        self.assertEqual(backend.compute_type, "int8")
        self.assertEqual(backend.language, "zh")
        self.assertEqual(backend.timeout_seconds, 9.5)

    def test_builds_default_asr_settings(self):
        with patch.dict("os.environ", {}, clear=True):
            backend = build_asr()

        self.assertEqual(backend.model_name, "base")
        self.assertEqual(backend.device, "cpu")
        self.assertEqual(backend.compute_type, "int8")
        self.assertEqual(backend.language, "zh")
        self.assertEqual(backend.timeout_seconds, 15.0)

    def test_builds_tts_from_environment(self):
        values = {
            "LINKDOG_TTS_VOICE": "zh-TW-HsiaoYuNeural",
            "LINKDOG_TTS_FALLBACK_VOICE": "Meijia",
            "LINKDOG_EDGE_TTS_COMMAND": "/tmp/edge-tts",
            "LINKDOG_FFMPEG_COMMAND": "/tmp/ffmpeg",
        }
        with patch.dict("os.environ", values, clear=True):
            backend = build_tts()

        self.assertIsInstance(backend, CommandTTSBackend)
        self.assertEqual(backend.voice, "zh-TW-HsiaoYuNeural")
        self.assertEqual(backend.edge_tts_command, "/tmp/edge-tts")
        self.assertEqual(backend.ffmpeg_command, "/tmp/ffmpeg")

    def test_default_tts_uses_edge_tts_from_active_python_environment(self):
        with patch.dict("os.environ", {}, clear=True):
            backend = build_tts()

        self.assertTrue(backend.edge_tts_command.endswith("/.venv-dev/bin/edge-tts"))

    def test_builds_singleton_pocket_tts_backend_from_environment(self):
        values = {
            "LINKDOG_TTS_BACKEND": "pocket",
            "LINKDOG_POCKET_VOICE": "cosette",
        }
        with (
            patch.dict("os.environ", values, clear=True),
            patch.object(main, "_POCKET_TTS_BACKEND", None),
        ):
            first = main.build_tts()
            second = main.build_tts()

        self.assertIsInstance(first, PocketTTSBackend)
        self.assertIs(first, second)
        self.assertEqual(first.voice, "cosette")

    def test_builds_16khz_60ms_opus_player(self):
        session = DeviceSession("TEST:DOG", FakeWebSocket())

        player = build_player(session)

        self.assertIsInstance(player, OpusDownlinkPlayer)
        self.assertEqual(player.codec.sample_rate, 16_000)
        self.assertEqual(player.codec.channels, 1)
        self.assertEqual(player.codec.frame_duration_ms, 60)
        player.codec.close()

    def test_voice_action_executor_binds_device_id(self):
        async def scenario():
            with patch("app.main.execute_voice_action", new=AsyncMock(
                return_value="Okay, I sat down."
            )) as execute:
                executor = build_voice_action_executor("TEST:DOG")
                response = await executor("sit_down")

            execute.assert_awaited_once_with("TEST:DOG", "sit_down")
            self.assertEqual(response, "Okay, I sat down.")

        asyncio.run(scenario())

    def test_voice_volume_executor_binds_device_id(self):
        async def scenario():
            arguments = {"mode": "up"}
            with patch("app.main.execute_voice_volume", new=AsyncMock(
                return_value="Volume set to 70 percent."
            )) as execute:
                executor = build_voice_volume_executor("TEST:DOG")
                response = await executor(arguments)

            execute.assert_awaited_once_with("TEST:DOG", arguments)
            self.assertEqual(response, "Volume set to 70 percent.")

        asyncio.run(scenario())

    def test_builds_hermes_client_from_environment(self):
        values = {
            "LINKDOG_HERMES_API_URL": "http://127.0.0.1:9999/v1",
            "LINKDOG_HERMES_API_KEY": "local-key",
            "LINKDOG_HERMES_PROVIDER": "ollama-cloud",
            "LINKDOG_HERMES_MODEL": "deepseek-v4-pro",
            "LINKDOG_HERMES_HISTORY_TURNS": "4",
            "LINKDOG_HERMES_TIMEOUT": "45",
        }
        with tempfile.TemporaryDirectory() as directory:
            empty_store = SettingsStore(Path(directory) / "settings.json")
            with (
                patch.dict("os.environ", values, clear=True),
                patch.object(main, "SETTINGS_STORE", empty_store),
            ):
                client = build_hermes_client()

        self.assertIsInstance(client, HermesAPIClient)
        self.assertEqual(client.base_url, "http://127.0.0.1:9999/v1")
        self.assertEqual(client.api_key, "local-key")
        self.assertEqual(client.model, "deepseek-v4-pro")
        self.assertEqual(client.provider, "ollama-cloud")
        self.assertEqual(client.max_history_turns, 4)
        self.assertEqual(client.timeout_seconds, 45.0)
        self.assertEqual(client.allowed_tool_actions, {
            "sit_down", "stand_up", "get_down", "shake_hands"
        })
        self.assertEqual(client.tools[0]["function"]["name"], "linkdog_action")
        self.assertEqual(
            {tool["function"]["name"] for tool in client.tools},
            {"linkdog_action", "linkdog_volume"},
        )
        action_schema = client.tools[0]["function"]["parameters"]["properties"]["action"]
        self.assertEqual(set(action_schema["enum"]), client.allowed_tool_actions)
        volume_tool = next(
            tool for tool in client.tools
            if tool["function"]["name"] == "linkdog_volume"
        )
        volume_properties = volume_tool["function"]["parameters"]["properties"]
        self.assertEqual(volume_properties["volume"]["minimum"], 10)
        self.assertEqual(volume_properties["volume"]["maximum"], 100)

    def test_routes_listen_start_to_pipeline(self):
        pipeline = Mock()
        pipeline.is_listening = False

        handle_device_event(
            "TEST:DOG",
            pipeline,
            {"type": "listen", "state": "start", "mode": "auto"},
        )

        pipeline.start_listening.assert_called_once_with()

    def test_does_not_restart_pipeline_when_already_listening(self):
        pipeline = Mock()
        pipeline.is_listening = True

        handle_device_event(
            "TEST:DOG",
            pipeline,
            {"type": "listen", "state": "start", "mode": "auto"},
        )

        pipeline.start_listening.assert_not_called()

    def test_does_not_restart_pipeline_for_listen_detect(self):
        pipeline = Mock()

        handle_device_event(
            "TEST:DOG",
            pipeline,
            {"type": "listen", "state": "detect", "text": "你好小智"},
        )

        pipeline.start_listening.assert_not_called()

    def test_routes_wake_word_abort_to_voice_turn(self):
        async def scenario():
            session = DeviceSession("TEST:DOG", FakeWebSocket())
            voice_turn = Mock()
            voice_turn.session = session
            voice_turn.abort = AsyncMock()
            try:
                task = handle_device_event(
                    "TEST:DOG",
                    Mock(),
                    {"type": "abort", "reason": "wake_word_detected"},
                    voice_turn=voice_turn,
                )
                await task
                voice_turn.abort.assert_awaited_once_with("wake_word_detected")
            finally:
                await session.close()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
