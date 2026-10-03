"""The model source *contract*: real precedence, and recovery when a model retires.

Second review (2026-09-14) noted two gaps in the first round of tests:

1. ``test_settings_json_still_overrides_the_env_fallback`` was named as though it
   exercised the loading precedence, but it only round-tripped a
   ``SettingsStore``. It never called ``load_dashboard_settings()`` at all, so
   nothing verified that ``settings.json`` actually beats ``.env``, nor that the
   environment is only a fallback.
2. Nothing covered what happens when the persisted model is **retired from the
   catalog**. That is the failure the fix was written for: the dashboard would
   keep returning a model it then refuses to persist (``PUT /api/settings``
   returns 422), leaving the operator stuck with no way back through the UI.

These tests call the real loader with conflicting values, so the precedence is
established by execution rather than by reading the code.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.dashboard_settings import DEFAULT_MODEL, DashboardSettings, SettingsStore  # noqa: E402


def _settings_path(directory: str) -> Path:
    return Path(directory) / "settings.json"


def _save(path: Path, model: str) -> SettingsStore:
    store = SettingsStore(path)
    store.save(
        DashboardSettings(
            agent_name="LinkDog",
            system_prompt="p",
            model=model,
            memory_enabled=True,
            max_history_turns=6,
            volume=70,
        )
    )
    return store


class ModelPrecedenceTests(unittest.TestCase):
    """``settings.json`` > environment > ``DEFAULT_MODEL``, by execution."""

    def test_settings_json_beats_a_conflicting_environment_value(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _save(_settings_path(directory), "glm-5.3")
            import app.main as main

            with (
                mock.patch.dict(
                    "os.environ", {"LINKDOG_HERMES_MODEL": "env-loses"}, clear=False
                ),
                mock.patch.object(main, "SETTINGS_STORE", store),
            ):
                settings = main.load_dashboard_settings()

        self.assertEqual(settings.model, "glm-5.3")

    def test_environment_value_is_used_when_no_settings_file_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(_settings_path(directory))  # never saved
            import app.main as main

            with (
                mock.patch.dict(
                    "os.environ", {"LINKDOG_HERMES_MODEL": "env-only"}, clear=False
                ),
                mock.patch.object(main, "SETTINGS_STORE", store),
            ):
                settings = main.load_dashboard_settings()

        self.assertEqual(settings.model, "env-only")

    def test_default_model_is_used_when_neither_source_provides_one(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(_settings_path(directory))  # never saved
            import app.main as main

            with (
                mock.patch.dict("os.environ", {}, clear=True),
                mock.patch.object(main, "SETTINGS_STORE", store),
            ):
                settings = main.load_dashboard_settings()

        self.assertEqual(settings.model, DEFAULT_MODEL)

    def test_the_built_client_uses_the_persisted_model(self):
        """The value that reaches the client is the one ``settings.json`` holds."""
        with tempfile.TemporaryDirectory() as directory:
            store = _save(_settings_path(directory), "glm-5.3")
            import app.main as main

            with (
                mock.patch.dict(
                    "os.environ",
                    {"LINKDOG_HERMES_MODEL": "env-loses", "LINKDOG_HERMES_API_KEY": "k"},
                    clear=False,
                ),
                mock.patch.object(main, "SETTINGS_STORE", store),
            ):
                client = main.build_hermes_client()

        self.assertEqual(client.model, "glm-5.3")


class RetiredModelRecoveryTests(unittest.TestCase):
    """When the persisted model leaves the catalog, there must be a way back.

    The original defect: the code default was ``deepseek-v4-flash:0731``, which
    Ollama no longer serves. ``PUT /api/settings`` validates against the live
    catalog and returns 422 for anything absent from it, so a dashboard showing
    that retired literal was showing a value the operator could never save.
    """

    def test_retired_default_is_absent_from_the_live_catalog_shape(self):
        """Documents the retired literal; the current default is a different id."""
        self.assertNotEqual(DEFAULT_MODEL, "deepseek-v4-flash:0731")
        self.assertEqual(DEFAULT_MODEL, "deepseek-v4.1-flash")

    def test_current_default_is_a_catalog_eligible_shape(self):
        """A default that cannot pass the catalog filter would re-create the bug.

        ``filter_highest_version_models`` keeps only the highest version per
        family, so a versioned-suffix id (``:0731``) is the shape that tends to
        disappear. The default must be an unversioned family id.
        """
        from app.model_catalog import parse_model_family_version

        self.assertNotIn(
            ":",
            DEFAULT_MODEL,
            "a versioned-suffix id is the shape that retires and strands the UI",
        )
        parsed = parse_model_family_version(DEFAULT_MODEL)
        self.assertIsNotNone(parsed, f"{DEFAULT_MODEL} should parse as a known family")

    def test_default_model_is_reachable_from_a_fresh_settings_file(self):
        """Recovery path: delete settings.json and the fallback is catalog-valid."""
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(_settings_path(directory))
            import app.main as main

            with (
                mock.patch.dict("os.environ", {}, clear=True),
                mock.patch.object(main, "SETTINGS_STORE", store),
            ):
                settings = main.load_dashboard_settings()

        # This is exactly what the operator gets after removing the file, and it
        # must be a model the dashboard will accept back.
        self.assertEqual(settings.model, DEFAULT_MODEL)

    def test_retired_model_in_settings_file_is_still_returned_verbatim(self):
        """The loader must not silently rewrite an operator's saved choice.

        Silently swapping the model would be a worse bug than the one being
        fixed: it would change behaviour without telling anyone. The contract is
        that loading preserves the stored value; the *default* (used only when
        no file exists) is what had to be modernised.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = _settings_path(directory)
            store = _save(path, "deepseek-v4-flash:0731")
            import app.main as main

            with mock.patch.object(main, "SETTINGS_STORE", store):
                settings = main.load_dashboard_settings()

        self.assertEqual(settings.model, "deepseek-v4-flash:0731")

    def test_settings_file_is_valid_json_and_round_trips(self):
        """A corrupt persisted file would break the dashboard on next boot."""
        with tempfile.TemporaryDirectory() as directory:
            path = _settings_path(directory)
            _save(path, "glm-5.3")

            payload = json.loads(path.read_text(encoding="utf-8"))
            # Shape is {"version": N, "settings": {...}} — not a flat object.
            self.assertEqual(payload["version"], 1)
            self.assertEqual(payload["settings"]["model"], "glm-5.3")
            self.assertEqual(SettingsStore(path).load().model, "glm-5.3")


if __name__ == "__main__":
    unittest.main()
