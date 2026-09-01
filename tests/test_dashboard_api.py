import tempfile
import unittest
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import app.main as main
from app.dashboard_settings import DashboardSettings, SettingsStore
from app.model_catalog import CatalogResult, CatalogUpstreamError


class DashboardApiTests(unittest.TestCase):
    def setUp(self):
        main.ACTIVE_SESSIONS.clear()
        self.directory = tempfile.TemporaryDirectory()
        defaults = DashboardSettings(
            agent_name="Xiaobin",
            system_prompt="Be a playful robot dog.",
            model="glm-5.3-flash",
            memory_enabled=True,
            max_history_turns=6,
            user_profile="",
            context_memory="",
            volume=70,
        )
        self.store = SettingsStore(
            Path(self.directory.name) / "settings.json",
            defaults=defaults,
        )
        self.store.save(defaults)
        self.store_patch = patch.object(main, "SETTINGS_STORE", self.store)
        self.store_patch.start()
        self.catalog = SimpleNamespace(get_models=AsyncMock(return_value=CatalogResult(
            models=(
                "deepseek-v4-flash:0731",
                "deepseek-v4-pro:0813",
                "glm-5.3",
                "glm-5.3-flash",
                "minimax-m3",
            ),
            stale=False,
        )))
        self.catalog_patch = patch.object(main, "MODEL_CATALOG", self.catalog)
        self.catalog_patch.start()
        self.client = TestClient(main.app)

    def tearDown(self):
        main.ACTIVE_SESSIONS.clear()
        self.catalog_patch.stop()
        self.store_patch.stop()
        self.directory.cleanup()

    def test_dashboard_is_served_from_adapter(self):
        response = self.client.get("/dashboard")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("LinkDog Studio", response.text)
        self.assertIn('<h1 id="page-title">LinkDog</h1>', response.text)
        self.assertNotIn('id="header-agent-name"', response.text)
        self.assertIn("Model &amp; memory", response.text)
        self.assertIn('id="device-mac"', response.text)
        self.assertIn('id="device-ip"', response.text)
        self.assertIn("MAC address", response.text)
        self.assertIn("IP address", response.text)
        self.assertIn('<select id="model" name="model" required>', response.text)
        self.assertNotIn('<datalist id="model-options">', response.text)
        self.assertNotIn('value="qwen3.5:cloud"', response.text)

    def test_dashboard_script_uses_live_catalog_and_handles_unavailable_model(self):
        response = self.client.get("/dashboard/assets/dashboard.js")

        self.assertEqual(response.status_code, 200)
        self.assertIn("fetch('/api/models')", response.text)
        self.assertIn("(unavailable)", response.text)
        self.assertIn("Model discovery failed", response.text)

    def test_get_models_returns_alphabetically_sorted_catalog_and_stale_state(self):
        catalog = SimpleNamespace(get_models=AsyncMock(return_value=CatalogResult(
            models=("minimax-m3", "glm-5.3", "deepseek-v4-flash:0731"),
            stale=True,
        )))

        with patch.object(main, "MODEL_CATALOG", catalog, create=True):
            response = self.client.get("/api/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "models": ["deepseek-v4-flash:0731", "glm-5.3", "minimax-m3"],
            "stale": True,
        })
        catalog.get_models.assert_awaited_once()

    def test_get_models_maps_uncached_upstream_failure_to_non_200(self):
        catalog = SimpleNamespace(get_models=AsyncMock(
            side_effect=CatalogUpstreamError("catalog unavailable")
        ))

        with patch.object(main, "MODEL_CATALOG", catalog, create=True):
            response = self.client.get("/api/models")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "catalog unavailable")

    def test_load_dashboard_settings_environment_fallback_uses_deepseek_flash(self):
        empty_store = SettingsStore(Path(self.directory.name) / "missing.json")
        with (
            patch.object(main, "SETTINGS_STORE", empty_store),
            patch.dict("os.environ", {}, clear=True),
        ):
            settings = main.load_dashboard_settings()

        self.assertEqual(settings.model, "deepseek-v4-flash:0731")

    def test_get_settings_returns_persisted_values_and_device_status(self):
        response = self.client.get("/api/settings")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["settings"]["agent_name"], "Xiaobin")
        self.assertEqual(data["settings"]["model"], "glm-5.3-flash")
        self.assertEqual(data["connected_devices"], [])
        self.assertEqual(data["connected_device_details"], [])

    def test_get_settings_returns_connected_device_mac_and_ip(self):
        main.ACTIVE_SESSIONS["TEST:DOG"] = SimpleNamespace(
            ip_address="10.1.1.42"
        )

        response = self.client.get("/api/settings")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["connected_devices"], ["TEST:DOG"])
        self.assertEqual(response.json()["connected_device_details"], [{
            "device_id": "TEST:DOG",
            "ip_address": "10.1.1.42",
        }])

    def test_put_settings_validates_and_persists_complete_payload(self):
        payload = {
            "agent_name": "Buddy",
            "system_prompt": "A sweet and playful young friend.",
            "model": "deepseek-v4-flash:0731",
            "memory_enabled": True,
            "max_history_turns": 8,
            "user_profile": "The user enjoys playful interaction.",
            "context_memory": "We are close friends.",
            "volume": 65,
        }

        response = self.client.put("/api/settings", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["settings"], payload)
        self.assertEqual(response.json()["connected_device_details"], [])
        self.assertEqual(self.store.load().agent_name, "Buddy")

    def test_put_settings_rejects_model_absent_from_catalog_without_rewriting_settings(self):
        before = self.store.load()
        payload = before.to_dict()
        payload["model"] = "gpt-oss:20b"

        response = self.client.put("/api/settings", json=payload)

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "model is not available")
        self.assertEqual(self.store.load(), before)

    def test_put_settings_catalog_outage_does_not_rewrite_settings(self):
        before = self.store.load()
        payload = before.to_dict()
        payload["agent_name"] = "Must not persist"
        self.catalog.get_models.side_effect = CatalogUpstreamError("catalog unavailable")

        response = self.client.put("/api/settings", json=payload)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "catalog unavailable")
        self.assertEqual(self.store.load(), before)

    def test_put_settings_rejects_invalid_volume(self):
        payload = self.store.load().to_dict()
        payload["volume"] = 9

        response = self.client.put("/api/settings", json=payload)

        self.assertEqual(response.status_code, 422)

    def test_put_settings_applies_volume_to_connected_device(self):
        payload = self.store.load().to_dict()
        payload["volume"] = 65
        main.ACTIVE_SESSIONS["TEST:DOG"] = object()

        with patch.object(
            main,
            "execute_voice_volume",
            new=AsyncMock(return_value="Volume set to 65 percent."),
        ) as execute:
            response = self.client.put("/api/settings", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["volume_applied_to"], ["TEST:DOG"])
        execute.assert_awaited_once_with(
            "TEST:DOG", {"mode": "set", "volume": 65}
        )

    def test_build_hermes_client_uses_dashboard_runtime_settings(self):
        self.store.save(DashboardSettings(
            agent_name="Buddy",
            system_prompt="A custom role.",
            model="qwen3.5:cloud",
            memory_enabled=False,
            max_history_turns=9,
            user_profile="User profile text.",
            context_memory="Context memory text.",
            volume=65,
        ))

        with patch.dict("os.environ", {"LINKDOG_HERMES_API_KEY": "test"}, clear=True):
            client = main.build_hermes_client()

        self.assertEqual(client.model, "qwen3.5:cloud")
        self.assertEqual(client.system_prompt, "A custom role.")
        self.assertEqual(client.max_history_turns, 0)

    def test_apply_saved_volume_uses_persisted_preference(self):
        self.store.save(DashboardSettings(
            agent_name="Xiaobin",
            system_prompt="Role.",
            model="glm-5.3-flash",
            memory_enabled=True,
            max_history_turns=6,
            volume=60,
        ))

        with patch.object(
            main,
            "execute_voice_volume",
            new=AsyncMock(return_value="Volume set to 60 percent."),
        ) as execute:
            asyncio.run(main.apply_saved_volume("TEST:DOG"))

        execute.assert_awaited_once_with(
            "TEST:DOG", {"mode": "set", "volume": 60}
        )


if __name__ == "__main__":
    unittest.main()
