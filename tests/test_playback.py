import asyncio
import json
import unittest

from app.device_session import DeviceSession, DeviceState
from app.playback import OpusDownlinkPlayer, PlaybackError


class FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def send_text(self, text):
        self.messages.append(json.loads(text))

    async def send_bytes(self, payload):
        self.messages.append(payload)


class FakeCodec:
    samples_per_channel = 960
    channels = 1

    def __init__(self, error=None):
        self.frames = []
        self.error = error

    def encode(self, frame):
        self.frames.append(frame)
        if self.error:
            raise self.error
        return b"opus-" + bytes([len(self.frames)])


class FakeSleep:
    def __init__(self):
        self.delays = []

    async def __call__(self, delay):
        self.delays.append(delay)


class OpusDownlinkPlayerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.websocket = FakeWebSocket()
        self.session = DeviceSession("TEST:DOG", self.websocket)
        self.codec = FakeCodec()
        self.sleep = FakeSleep()
        self.player = OpusDownlinkPlayer(
            self.session,
            self.codec,
            frame_duration_ms=60,
            sleep=self.sleep,
        )

    async def asyncTearDown(self):
        await self.session.close()

    async def test_sends_metadata_audio_and_stop_in_protocol_order(self):
        # 7 frames: first 5 are pre-buffered (sent immediately), last 2 paced.
        pcm = b"\x01\x00" * (960 * 7)

        await self.player.play("太好了！", pcm)

        self.assertEqual(self.websocket.messages, [
            {"type": "tts", "state": "start"},
            {"type": "llm", "emotion": "happy"},
            {"type": "tts", "state": "sentence_start", "text": "太好了！"},
            b"opus-\x01",
            b"opus-\x02",
            b"opus-\x03",
            b"opus-\x04",
            b"opus-\x05",
            b"opus-\x06",
            b"opus-\x07",
            {"type": "tts", "state": "stop"},
        ])
        self.assertEqual(len(self.codec.frames), 7)
        self.assertEqual(len(self.codec.frames[-1]), 960 * 2)
        self.assertEqual(self.sleep.delays, [0.06, 0.06])
        self.assertEqual(self.session.state, DeviceState.LISTENING)

    async def test_explicit_unknown_emotion_falls_back_to_neutral(self):
        await self.player.play("收到。", b"\x00\x00" * 960, emotion="not-real")

        self.assertEqual(
            self.websocket.messages[1],
            {"type": "llm", "emotion": "neutral"},
        )

    async def test_encode_failure_still_sends_stop_and_recovers(self):
        player = OpusDownlinkPlayer(
            self.session,
            FakeCodec(error=RuntimeError("encode failed")),
            sleep=self.sleep,
        )

        with self.assertRaisesRegex(PlaybackError, "encode failed"):
            await player.play("回答", b"\x00\x00" * 960)

        self.assertEqual(self.websocket.messages[-1], {"type": "tts", "state": "stop"})
        self.assertEqual(self.session.state, DeviceState.LISTENING)

    async def test_send_failure_attempts_recovery_without_masking_error(self):
        class FailingWebSocket(FakeWebSocket):
            async def send_bytes(self, payload):
                raise RuntimeError("socket failed")

        session = DeviceSession("FAIL:DOG", FailingWebSocket())
        player = OpusDownlinkPlayer(session, FakeCodec(), sleep=self.sleep)
        try:
            with self.assertRaisesRegex(PlaybackError, "socket failed"):
                await player.play("回答", b"\x00\x00" * 960)
            self.assertEqual(session.state, DeviceState.LISTENING)
        finally:
            await session.close()

    async def test_stalled_audio_send_times_out_and_releases_turn_lock(self):
        class HangingWebSocket(FakeWebSocket):
            async def send_bytes(self, payload):
                await asyncio.Event().wait()

        websocket = HangingWebSocket()
        session = DeviceSession("HANG:DOG", websocket)
        session.state = DeviceState.THINKING
        player = OpusDownlinkPlayer(
            session,
            FakeCodec(),
            sleep=self.sleep,
            playback_timeout_seconds=0.01,
        )
        try:
            with self.assertRaisesRegex(PlaybackError, "timed out"):
                await asyncio.wait_for(
                    player.play("Okay, I sat down.", b"\x00\x00" * 960),
                    timeout=1,
                )

            self.assertEqual(
                websocket.messages[-1],
                {"type": "tts", "state": "stop"},
            )
            self.assertEqual(session.state, DeviceState.LISTENING)
        finally:
            await session.close()

    async def test_streaming_begin_feed_finish_sends_start_stop_once(self):
        await self.player.begin("第一句。")
        await self.player.feed(b"\x01\x00" * 960)
        await self.player.feed(b"\x02\x00" * 960)
        await self.player.finish()

        self.assertEqual(self.websocket.messages, [
            {"type": "tts", "state": "start"},
            {"type": "llm", "emotion": "neutral"},
            {"type": "tts", "state": "sentence_start", "text": "第一句。"},
            b"opus-\x01",
            b"opus-\x02",
            {"type": "tts", "state": "stop"},
        ])
        self.assertEqual(self.session.state, DeviceState.LISTENING)

    async def test_begin_reuses_endpoint_start_for_locked_turn(self):
        self.session.state = DeviceState.THINKING

        await self.player.begin("First sentence.")
        await self.player.feed(b"\x01\x00" * 960)
        await self.player.finish()

        self.assertEqual(self.websocket.messages, [
            {"type": "llm", "emotion": "neutral"},
            {
                "type": "tts",
                "state": "sentence_start",
                "text": "First sentence.",
            },
            b"opus-\x01",
            {"type": "tts", "state": "stop"},
        ])
        self.assertEqual(self.session.state, DeviceState.LISTENING)

    async def test_feed_before_begin_raises(self):
        with self.assertRaisesRegex(PlaybackError, "before begin"):
            await self.player.feed(b"\x00\x00" * 960)


if __name__ == "__main__":
    unittest.main()
