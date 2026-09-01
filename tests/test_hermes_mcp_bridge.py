import json
import unittest

import httpx

from app.hermes_tools import LinkDogClient, LinkDogToolError


class LinkDogClientTests(unittest.TestCase):
    def _client(self, handler):
        return LinkDogClient(
            adapter_url="http://adapter.test",
            device_id="TEST:DOG",
            transport=httpx.MockTransport(handler),
        )

    def test_sit_calls_safe_adapter_action_and_requires_completed(self):
        def handler(request):
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.path, "/xiaozhi/action")
            self.assertEqual(json.loads(request.content), {
                "device_id": "TEST:DOG",
                "action": "sit_down",
            })
            return httpx.Response(200, json={
                "status": "completed",
                "device_id": "TEST:DOG",
                "action": "sit_down",
                "request_id": 10,
            })

        result = self._client(handler).execute("sit_down")
        self.assertEqual(result["status"], "completed")

    def test_rejects_unknown_action_without_network_call(self):
        called = False

        def handler(request):
            nonlocal called
            called = True
            return httpx.Response(500)

        with self.assertRaises(LinkDogToolError):
            self._client(handler).execute("not_a_real_action")
        self.assertFalse(called)

    def test_rejects_angle_without_part_without_network_call(self):
        called = False

        def handler(request):
            nonlocal called
            called = True
            return httpx.Response(500)

        with self.assertRaises(LinkDogToolError):
            self._client(handler).execute("angle", angle=90)
        self.assertFalse(called)

    def test_surfaces_adapter_failure_without_claiming_completion(self):
        def handler(request):
            return httpx.Response(504, json={"detail": "device action timed out"})

        with self.assertRaisesRegex(LinkDogToolError, "timed out"):
            self._client(handler).execute("stand_up")

    def test_status_reports_adapter_and_connected_device(self):
        def handler(request):
            self.assertEqual(request.url.path, "/health")
            return httpx.Response(200, json={
                "status": "ok",
                "connected_devices": ["TEST:DOG"],
            })

        result = self._client(handler).status()
        self.assertTrue(result["connected"])
        self.assertEqual(result["device_id"], "TEST:DOG")


if __name__ == "__main__":
    unittest.main()
