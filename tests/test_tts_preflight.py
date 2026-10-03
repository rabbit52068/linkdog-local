"""Preflight decision tests: fail-closed only for the voice-cloning path."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SPEC_PATH = _PROJECT_ROOT / "scripts" / "tts_preflight.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("tts_preflight", _SPEC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = _load_module()


class VoiceClassificationTests(unittest.TestCase):
    def test_catalog_voice_is_not_a_path(self):
        self.assertEqual(preflight.classify_voice("cosette"), "catalog")
        self.assertEqual(preflight.classify_voice("alba"), "catalog")

    def test_empty_voice_resolves_to_catalog_fallback(self):
        # Empty voice is normalised to the default catalog voice, which is a
        # catalog kind (not "empty"/"path"), matching runtime fallback.
        self.assertEqual(preflight.resolve_voice({"LINKDOG_POCKET_VOICE": ""}), "cosette")
        self.assertEqual(
            preflight.classify_voice(
                preflight.resolve_voice({"LINKDOG_POCKET_VOICE": ""})
            ),
            "catalog",
        )

    def test_file_path_is_audio(self):
        self.assertEqual(
            preflight.classify_voice("/tmp/butterfly.wav"), "audio"
        )
        self.assertEqual(
            preflight.classify_voice("voices/butterfly.wav"), "audio"
        )

    def test_unknown_name_is_audio(self):
        # A non-catalog, non-empty value is an audio reference (fails loudly,
        # never silently falls back to a catalog voice).
        self.assertEqual(preflight.classify_voice("not_a_voice"), "audio")

    def test_safetensors_is_state_not_audio(self):
        self.assertEqual(preflight.classify_voice("/tmp/v.safetensors"), "state")
        self.assertEqual(
            preflight.classify_voice("hf://kyutai/tts-voices/x.safetensors"), "state"
        )
        self.assertEqual(
            preflight.classify_voice(
                "https://huggingface.co/x/resolve/main/y.safetensors"
            ),
            "state",
        )


class ResolveBackendTests(unittest.TestCase):
    def test_backend_normalises_case_and_space(self):
        self.assertEqual(preflight.resolve_backend({"LINKDOG_TTS_BACKEND": " pocket "}), "pocket")
        self.assertEqual(preflight.resolve_backend({"LINKDOG_TTS_BACKEND": "Pocket"}), "pocket")
        self.assertEqual(preflight.resolve_backend({"LINKDOG_TTS_BACKEND": "edge"}), "edge")

    def test_backend_defaults_to_command(self):
        self.assertEqual(preflight.resolve_backend({}), "command")
        self.assertEqual(preflight.resolve_backend({"LINKDOG_TTS_BACKEND": "   "}), "command")


class ResolveVoiceTests(unittest.TestCase):
    def test_empty_and_blank_fall_back_to_cosette(self):
        self.assertEqual(preflight.resolve_voice({"LINKDOG_POCKET_VOICE": ""}), "cosette")
        self.assertEqual(preflight.resolve_voice({"LINKDOG_POCKET_VOICE": "   "}), "cosette")
        self.assertEqual(preflight.resolve_voice({}), "cosette")

    def test_case_and_space_normalise_to_catalog(self):
        self.assertEqual(preflight.resolve_voice({"LINKDOG_POCKET_VOICE": "COSETTE"}), "cosette")
        self.assertEqual(preflight.resolve_voice({"LINKDOG_POCKET_VOICE": " cosette "}), "cosette")

    def test_path_preserved(self):
        self.assertEqual(
            preflight.resolve_voice({"LINKDOG_POCKET_VOICE": "/tmp/butterfly.wav"}),
            "/tmp/butterfly.wav",
        )


class RunTests(unittest.TestCase):
    def test_catalog_voice_warns_and_returns_zero(self):
        env = {"LINKDOG_TTS_BACKEND": "pocket", "LINKDOG_POCKET_VOICE": "cosette"}
        out = []
        code = preflight.run(model_factory=lambda: None, env=env, out=out)
        self.assertEqual(code, 0)
        self.assertTrue(any("WARNING" in line for line in out))

    def test_missing_audio_path_aborts_nonzero(self):
        env = {
            "LINKDOG_TTS_BACKEND": "pocket",
            "LINKDOG_POCKET_VOICE": "/tmp/does-not-exist-12345.wav",
        }
        out = []
        code = preflight.run(model_factory=lambda: None, env=env, out=out)
        self.assertNotEqual(code, 0)
        self.assertTrue(any("FATAL" in line for line in out))

    def test_audio_without_cloning_aborts_nonzero(self):
        import tempfile

        model = SimpleNamespace(has_voice_cloning=False)

        def factory():
            return model

        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            env = {
                "LINKDOG_TTS_BACKEND": "pocket",
                "LINKDOG_POCKET_VOICE": handle.name,
            }
            out = []
            code = preflight.run(model_factory=factory, env=env, out=out)
        self.assertNotEqual(code, 0)
        self.assertTrue(any("voice cloning" in line for line in out))

    def test_backend_with_whitespace_is_treated_as_pocket(self):
        # Regression B1: ' pocket ' must behave exactly like 'pocket'.
        env = {
            "LINKDOG_TTS_BACKEND": " pocket ",
            "LINKDOG_POCKET_VOICE": "cosette",
        }
        out = []
        code = preflight.run(model_factory=lambda: None, env=env, out=out)
        self.assertEqual(code, 0)
        self.assertTrue(any("catalog voice" in line for line in out))

    def test_non_pocket_backend_skips_without_model(self):
        # Regression B4: edge + a stale local voice must return 0 and never
        # call the model factory.
        import tempfile

        calls = []

        def factory():
            calls.append(True)
            return None

        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            env = {
                "LINKDOG_TTS_BACKEND": "edge",
                "LINKDOG_POCKET_VOICE": handle.name,
            }
            out = []
            code = preflight.run(model_factory=factory, env=env, out=out)
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        self.assertTrue(any("skipped" in line for line in out))

    def test_safetensors_state_does_not_fail_closed_without_cloning(self):
        # Regression B5: .safetensors is imported directly, no cloning needed,
        # so a model without voice cloning must NOT abort here.
        import tempfile

        import numpy as np

        model = SimpleNamespace(has_voice_cloning=False)

        def get_state(voice):
            return {"voice": voice}

        def generate_audio_stream(state, text, copy_state=True):
            yield np.asarray([0.1, 0.2], dtype=np.float32)

        model.get_state_for_audio_prompt = get_state
        model.generate_audio_stream = generate_audio_stream

        calls = []

        def factory():
            calls.append(True)
            return model

        with tempfile.NamedTemporaryFile(suffix=".safetensors") as handle:
            env = {
                "LINKDOG_TTS_BACKEND": "pocket",
                "LINKDOG_POCKET_VOICE": handle.name,
            }
            out = []
            code = preflight.run(model_factory=factory, env=env, out=out)
        self.assertEqual(code, 0)
        self.assertEqual(calls, [True])
        self.assertTrue(any("state" in line for line in out))


class PreflightOutputSafetyTests(unittest.TestCase):
    """Second review: preflight output leaked raw text and implied token checks.

    Two separate problems, both fixed in the same place.

    1. ``_emit`` printed whatever it was given. The messages interpolate a
       configured voice path and raw exception text, and Hub errors embed the
       request URL — which can carry ``?token=...``. The project already had
       ``app.redact.redact_secrets``; preflight simply never used it.
    2. The script returned 0 with no mention of credentials, so a green run read
       as "the token is fine". It did not check the token at all: a cached model
       needs no credential, so preflight can pass with no token present.
    """

    def test_redaction_masks_a_token_embedded_in_an_error_message(self):
        from app.redact import redact_secrets

        raw = (
            "[tts_preflight] FATAL: could not load Pocket TTS model: "
            "HTTP 401 for url https://huggingface.co/api/models/x"
            "?token=hf_LEAKCANARY0000000000"
        )
        safe = redact_secrets(raw)

        self.assertNotIn("hf_LEAKCANARY0000000000", safe)
        # The message must stay diagnosable — only the value is removed.
        self.assertIn("HTTP 401", safe)
        self.assertIn("huggingface.co", safe)

    def test_emit_redacts_whatever_it_is_handed(self):
        out = []
        preflight._emit(
            out,
            "Authorization: Bearer abcdefghijklmnop and key sk-abcdefghijklmnop",
        )

        self.assertEqual(len(out), 1)
        self.assertNotIn("abcdefghijklmnop", out[0])
        self.assertIn("[REDACTED]", out[0])

    def test_emit_redacts_on_the_list_seam_too(self):
        """Both output paths must be covered, not just the file-like one."""
        out = []
        preflight._emit(out, "token=hf_LEAKCANARY0000000000")
        self.assertNotIn("hf_LEAKCANARY0000000000", out[0])

    def test_a_ready_run_reports_credential_durability_separately(self):
        """The green path must still say something about the credential."""
        import tempfile

        import numpy as np
        from types import SimpleNamespace

        model = SimpleNamespace(
            has_voice_cloning=True,
            get_state_for_audio_prompt=lambda voice: {"v": voice},
        )
        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            env = {
                "LINKDOG_TTS_BACKEND": "pocket",
                "LINKDOG_POCKET_VOICE": handle.name,
            }
            out = []

            def factory():
                return model

            def fake_synth(model_, state, text="x"):
                return np.ones(64, dtype=np.float32)

            original = preflight.synthesize_short
            preflight.synthesize_short = fake_synth
            try:
                code = preflight.run(model_factory=factory, env=env, out=out)
            finally:
                preflight.synthesize_short = original

        self.assertEqual(code, 0)
        credential_lines = [line for line in out if "credential:" in line]
        self.assertEqual(
            len(credential_lines),
            1,
            f"a ready run must report credential provenance once: {out}",
        )

    def test_readiness_does_not_assert_credential_health(self):
        """A non-durable credential must not be silently presented as fine."""
        from app.redact import redact_secrets

        durability, detail = preflight._credential_provenance()
        self.assertIn(durability, {"durable", "not_durable", "unknown"})
        # The earlier assertion here was `assertNotIn("hf_", detail)`, which is
        # simply wrong: the message names the credential path
        # (`.../secrets/hf_token`), so it legitimately contains `hf_`. What
        # actually must never appear is credential *material*, so run the
        # project's own redactor over it and require it to be a no-op.
        self.assertEqual(
            redact_secrets(detail),
            detail,
            f"provenance detail carries credential-shaped material: {detail}",
        )


if __name__ == "__main__":
    unittest.main()
