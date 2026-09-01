import json
import tempfile
import unittest
from pathlib import Path

from app.dashboard_settings import DashboardSettings, SettingsStore


class SettingsStoreTests(unittest.TestCase):
    def test_dashboard_settings_default_model_is_deepseek_flash(self):
        self.assertEqual(DashboardSettings().model, "deepseek-v4-flash:0731")

    def test_missing_file_uses_supplied_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            store = SettingsStore(
                path,
                defaults=DashboardSettings(
                    agent_name="Xiaobin",
                    system_prompt="Be playful.",
                    model="glm-5.3-flash",
                    memory_enabled=True,
                    max_history_turns=6,
                    user_profile="",
                    context_memory="",
                    volume=70,
                ),
            )

            settings = store.load()

            self.assertEqual(settings.agent_name, "Xiaobin")
            self.assertEqual(settings.model, "glm-5.3-flash")
            self.assertEqual(settings.volume, 70)
            self.assertFalse(path.exists())

    def test_save_is_persisted_as_versioned_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            store = SettingsStore(path)
            settings = DashboardSettings(
                agent_name="Buddy",
                system_prompt="A cheerful young friend.",
                model="qwen3.5:cloud",
                memory_enabled=True,
                max_history_turns=8,
                user_profile="The user likes stories.",
                context_memory="We are planning a picnic.",
                volume=65,
            )

            saved = store.save(settings)
            reloaded = store.load()
            payload = json.loads(path.read_text())

            self.assertEqual(saved, settings)
            self.assertEqual(reloaded, settings)
            self.assertEqual(payload["version"], 1)
            self.assertEqual(payload["settings"]["agent_name"], "Buddy")

    def test_rejects_out_of_range_volume(self):
        with self.assertRaises(ValueError):
            DashboardSettings(volume=101)

    def test_rejects_blank_model(self):
        with self.assertRaises(ValueError):
            DashboardSettings(model="   ")


if __name__ == "__main__":
    unittest.main()
