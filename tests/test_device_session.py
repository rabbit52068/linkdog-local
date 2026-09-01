import asyncio
import json
import unittest

from app.device_session import DeviceSession, DeviceState, SessionClosedError


class RecordingWebSocket:
    def __init__(self):
        self.messages = []
        self.active_sends = 0
        self.max_active_sends = 0
        self.close_calls = []

    async def _record(self, kind, value):
        self.active_sends += 1
        self.max_active_sends = max(self.max_active_sends, self.active_sends)
        await asyncio.sleep(0)
        self.messages.append((kind, value))
        self.active_sends -= 1

    async def send_text(self, text):
        await self._record("text", json.loads(text))

    async def send_bytes(self, payload):
        await self._record("bytes", payload)

    async def close(self, code=1000, reason=None):
        self.close_calls.append((code, reason))


class DeviceSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.websocket = RecordingWebSocket()
        self.session = DeviceSession(
            device_id="TEST:DOG",
            websocket=self.websocket,
            audio_queue_size=2,
        )

    async def asyncTearDown(self):
        await self.session.close()

    async def test_audio_queue_drops_oldest_frame_at_capacity(self):
        self.session.enqueue_audio(b"one")
        self.session.enqueue_audio(b"two")
        self.session.enqueue_audio(b"three")

        self.assertEqual(self.session.dropped_audio_frames, 1)
        self.assertEqual(await self.session.next_audio(), b"two")
        self.assertEqual(await self.session.next_audio(), b"three")

    async def test_json_and_audio_share_a_single_writer(self):
        await asyncio.gather(
            self.session.send_json({"type": "tts", "state": "start"}),
            self.session.send_audio(b"opus"),
        )

        self.assertEqual(self.websocket.max_active_sends, 1)
        self.assertCountEqual(
            self.websocket.messages,
            [
                ("text", {"type": "tts", "state": "start"}),
                ("bytes", b"opus"),
            ],
        )

    async def test_close_cancels_registered_tasks(self):
        started = asyncio.Event()

        async def worker():
            started.set()
            await asyncio.Event().wait()

        task = self.session.start_task(worker())
        await started.wait()
        await self.session.close()

        self.assertTrue(task.cancelled())
        self.assertEqual(self.session.state, DeviceState.CLOSED)

    async def test_closed_session_rejects_new_output(self):
        await self.session.close()

        with self.assertRaises(SessionClosedError):
            await self.session.send_audio(b"late")

if __name__ == "__main__":
    unittest.main()
