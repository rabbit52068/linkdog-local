import os
import unittest
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

PROTECTED = [
    ("get", "/api/health"),
    ("get", "/api/settings"),
    ("get", "/api/models"),
    ("post", "/xiaozhi/action"),
]


@pytest.mark.real_auth
class TokenAuthTests(unittest.TestCase):
    def client(self, host="10.0.0.9"):
        return TestClient(app, client=(host, 50000))

    def test_protected_routes_reject_missing_and_wrong_token(self):
        with patch.dict(os.environ, {"LINKDOG_API_TOKEN": "s3cret"}):
            client = self.client()
            for method, path in PROTECTED:
                response = getattr(client, method)(path)
                self.assertEqual(response.status_code, 401, path)
                response = getattr(client, method)(
                    path, headers={"Authorization": "Bearer wrong"}
                )
                self.assertEqual(response.status_code, 401, path)

    def test_correct_token_passes_auth(self):
        with patch.dict(os.environ, {"LINKDOG_API_TOKEN": "s3cret"}):
            response = self.client().get(
                "/api/health", headers={"Authorization": "Bearer s3cret"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("tts", response.json())

    def test_without_token_only_loopback_is_allowed(self):
        with patch.dict(os.environ, {"LINKDOG_API_TOKEN": ""}):
            self.assertEqual(
                self.client("10.0.0.9").get("/api/health").status_code, 403
            )
            self.assertEqual(
                self.client("127.0.0.1").get("/api/health").status_code, 200
            )

    def test_public_health_needs_no_token_and_hides_diagnostics(self):
        with patch.dict(os.environ, {"LINKDOG_API_TOKEN": "s3cret"}):
            response = self.client().get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("tts", response.json())

    def test_loopback_is_trusted_even_with_a_token_configured(self):
        with patch.dict(os.environ, {"LINKDOG_API_TOKEN": "s3cret"}):
            response = self.client("127.0.0.1").get("/api/health")
        self.assertEqual(response.status_code, 200)

    def test_device_routes_stay_open(self):
        with patch.dict(os.environ, {"LINKDOG_API_TOKEN": "s3cret"}):
            client = self.client()
            self.assertEqual(client.get("/xiaozhi/music/list.json").status_code, 200)
            self.assertEqual(
                client.get("/xiaozhi/ota/esp32s3/firmware.json").status_code, 200
            )
            self.assertEqual(client.post("/xiaozhi/ota/", content=b"{}").status_code, 200)
