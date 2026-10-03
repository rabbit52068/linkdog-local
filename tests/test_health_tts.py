"""Tests for /health TTS status exposure and the TTS-failure logging path."""

import asyncio
import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import app.main as main
import app.voice_turn as voice_turn
from app.asr import ASRError
from app.device_session import DeviceSession, DeviceState
from app.tts import TTSError


class FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def send_text(self, text):
        self.messages.append(text)

    async def send_bytes(self, payload):
        self.messages.append(payload)


class FakeVoiceInput:
    def __init__(self):
        self.utterances = asyncio.Queue()

    async def next_utterance(self):
        return await self.utterances.get()

    def start_listening(self):
        return True


class FakeASR:
    async def transcribe(self, pcm, sample_rate):
        return "你好"


class FakeHermes:
    async def complete(self, device_id, text):
        return "回答"

    async def close(self):
        pass


class FakeTTS:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    async def synthesize(self, text):
        self.calls.append(text)
        if self.error:
            raise self.error
        return b"pcm"


class FakePlayer:
    async def play(self, text, pcm, **kwargs):
        pass


class HealthTTSTests(unittest.TestCase):
    def setUp(self):
        main.ACTIVE_SESSIONS.clear()
        self.client = TestClient(main.app)

    def tearDown(self):
        main.ACTIVE_SESSIONS.clear()

    def test_health_unloaded_model_reports_unknown_not_false(self):
        # Regression B3 / test #4: before any load, "unobserved" must be None
        # (voice_cloning_available / configured_ok) and model_status 'unknown',
        # NOT false/false (the v1 false negative).
        values = {
            "LINKDOG_TTS_BACKEND": "pocket",
            "LINKDOG_POCKET_VOICE": "/tmp/some-voice.wav",
        }
        with (
            patch.dict("os.environ", values, clear=True),
            patch.object(main, "_POCKET_TTS_BACKEND", None),
        ):
            response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("connected_devices", data)
        tts = data["tts"]
        self.assertEqual(tts["backend"], "pocket")
        self.assertEqual(tts["voice"], "/tmp/some-voice.wav")
        self.assertEqual(tts["voice_kind"], "audio")
        self.assertTrue(tts["voice_cloning_required"])
        self.assertIsNone(tts["voice_cloning_available"])
        self.assertIsNone(tts["configured_ok"])
        self.assertEqual(tts["model_status"], "unknown")

    def test_health_catalog_voice_does_not_require_cloning(self):
        values = {
            "LINKDOG_TTS_BACKEND": "pocket",
            "LINKDOG_POCKET_VOICE": "cosette",
        }
        with (
            patch.dict("os.environ", values, clear=True),
            patch.object(main, "_POCKET_TTS_BACKEND", None),
        ):
            response = self.client.get("/api/health")

        data = response.json()["tts"]
        self.assertEqual(data["backend"], "pocket")
        self.assertEqual(data["voice"], "cosette")
        self.assertEqual(data["voice_kind"], "catalog")
        self.assertFalse(data["voice_cloning_required"])
        self.assertTrue(data["configured_ok"])

    def test_health_catalog_voice_normalises_case(self):
        # Regression B2/#3: 'COSETTE' must resolve to catalog and be ok.
        values = {
            "LINKDOG_TTS_BACKEND": "pocket",
            "LINKDOG_POCKET_VOICE": "COSETTE",
        }
        with (
            patch.dict("os.environ", values, clear=True),
            patch.object(main, "_POCKET_TTS_BACKEND", None),
        ):
            response = self.client.get("/api/health")

        data = response.json()["tts"]
        self.assertEqual(data["voice"], "cosette")
        self.assertEqual(data["voice_kind"], "catalog")
        self.assertTrue(data["configured_ok"])

    def test_health_model_ready_but_state_failed_reports_failed(self):
        # Regression B3/#5 (false positive): a model that exists but whose
        # state prompt failed must NOT report configured_ok=True.
        backend = SimpleNamespace(
            _model=SimpleNamespace(has_voice_cloning=True),
            _state=None,
            load_status="failed",
            last_error="get_state_for_audio_prompt exploded",
        )
        values = {
            "LINKDOG_TTS_BACKEND": "pocket",
            "LINKDOG_POCKET_VOICE": "/tmp/some-voice.wav",
        }
        with (
            patch.dict("os.environ", values, clear=True),
            patch.object(main, "_POCKET_TTS_BACKEND", backend),
        ):
            response = self.client.get("/api/health")

        tts = response.json()["tts"]
        self.assertEqual(tts["model_status"], "failed")
        self.assertIsNotNone(tts["last_error"])
        self.assertIsNot(tts["configured_ok"], True)

    def test_health_reflects_loaded_model_cloning_availability(self):
        backend = SimpleNamespace(
            _model=SimpleNamespace(has_voice_cloning=True),
            _state={"voice": "/tmp/some-voice.wav"},
            load_status="ready",
            last_error=None,
        )
        values = {
            "LINKDOG_TTS_BACKEND": "pocket",
            "LINKDOG_POCKET_VOICE": "/tmp/some-voice.wav",
        }
        with (
            patch.dict("os.environ", values, clear=True),
            patch.object(main, "_POCKET_TTS_BACKEND", backend),
        ):
            response = self.client.get("/api/health")

        tts = response.json()["tts"]
        self.assertEqual(tts["model_status"], "ready")
        self.assertTrue(tts["voice_cloning_available"])
        self.assertTrue(tts["configured_ok"])

    def test_health_command_backend_reports_configured_ok(self):
        with patch.dict("os.environ", {"LINKDOG_TTS_BACKEND": ""}, clear=True):
            response = self.client.get("/api/health")

        tts = response.json()["tts"]
        self.assertEqual(tts["backend"], "command")
        self.assertFalse(tts["voice_cloning_required"])
        self.assertTrue(tts["configured_ok"])
        self.assertEqual(tts["model_status"], "n/a")


class TTSFailureTotalTests(unittest.TestCase):
    def test_tts_failure_total_accumulates_across_workers(self):
        # Two independent workers over separate event loops must both increment
        # the process-wide counter, and the total must rise by 2.
        start = voice_turn.tts_failure_total()

        async def run_one_worker():
            websocket = FakeWebSocket()
            session = DeviceSession("TEST:DOG", websocket)
            voice_input = FakeVoiceInput()
            worker = voice_turn.VoiceTurnWorker(
                session,
                voice_input,
                FakeASR(),
                hermes=FakeHermes(),
                tts=FakeTTS(error=TTSError("offline")),
                player=FakePlayer(),
            )
            session.start_task(worker.run())
            await voice_input.utterances.put(b"pcm")
            await asyncio.wait_for(worker.wait_until_idle(), timeout=0.5)
            await session.close()

        asyncio.run(run_one_worker())
        asyncio.run(run_one_worker())

        self.assertEqual(voice_turn.tts_failure_total(), start + 2)
        self.assertIsNotNone(voice_turn.tts_failure_last())

    def test_tts_failure_logs_cause(self):
        # Regression B6 / #8: the exception __cause__ must be visible in the
        # warning log.
        captured = []

        class CaptureHandler(logging.Handler):
            def emit(self, record):
                captured.append(record.getMessage())

        handler = CaptureHandler()
        handler.setLevel(logging.WARNING)
        voice_turn.LOGGER.addHandler(handler)
        try:
            websocket = FakeWebSocket()
            session = DeviceSession("TEST:DOG", websocket)
            worker = voice_turn.VoiceTurnWorker(
                session, FakeVoiceInput(), FakeASR(), hermes=None, tts=None, player=None
            )

            inner = ValueError("ROOT_CAUSE_SENTINEL")
            outer = TTSError("synthesis failed")
            outer.__cause__ = inner
            worker._note_tts_failure(outer, "hello", 1)

            joined = "\n".join(captured)
            self.assertIn("ROOT_CAUSE_SENTINEL", joined)
            self.assertIn("ValueError", joined)
            self.assertIn("cause=", joined)
        finally:
            voice_turn.LOGGER.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
