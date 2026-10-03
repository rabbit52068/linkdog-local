"""Regression tests for the five open defects of the TTS guard v2.

Each test here fails against the pre-fix code and passes after it. They are
deliberately written at the *wiring* level (real ``build_tts`` / real
``/health`` / real logger) rather than as unit tests of the helpers, because
the original defects were precisely that the helpers were correct but nothing
asserted the runtime actually used them.

Covered:
  1. ``build_tts`` must route through ``tts_config`` normalisation.
  2. preflight must not apply a local ``Path.exists()`` to remote sources.
  3. ``_format_cause`` must keep the ROOT cause visible when the outer
     exception message is long.
  4. credentials must never reach ``/health`` or the TTS failure log.
  5. a ``failed`` model load must not report ``configured_ok=True`` for a
     catalog voice.
"""

import importlib.util
import json
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import app.main as main
import app.voice_turn as voice_turn
from app.pocket_tts import PocketTTSBackend
from app.redact import redact_secrets
from app.tts import TTSError

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

SYNTHETIC_SECRET = "SK-SYNTHETIC-SECRET-MARKER"


def _load_preflight():
    spec = importlib.util.spec_from_file_location(
        "tts_preflight_defects", _PROJECT_ROOT / "scripts" / "tts_preflight.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = _load_preflight()


# --------------------------------------------------------------------------
# Defect 1: build_tts() wiring
# --------------------------------------------------------------------------


class BuildTTSWiringTests(unittest.TestCase):
    """``build_tts`` must consume the single source of truth, not raw env.

    The pre-fix mutant read ``os.environ`` directly inside ``build_tts`` and
    all 82 related tests still passed, because only the *helpers* were covered.
    These tests call the real ``build_tts`` with values that require
    normalisation, so that mutant now fails.
    """

    def test_build_tts_normalises_backend_and_voice(self):
        values = {
            "LINKDOG_TTS_BACKEND": "  Pocket  ",
            "LINKDOG_POCKET_VOICE": "  COSETTE  ",
        }
        with (
            patch.dict("os.environ", values, clear=True),
            patch.object(main, "_POCKET_TTS_BACKEND", None),
        ):
            backend = main.build_tts()

        self.assertIsInstance(backend, PocketTTSBackend)
        # '  COSETTE  ' must arrive canonicalised; a raw-env mutant yields
        # '  COSETTE  ' here and fails.
        self.assertEqual(backend.voice, "cosette")

    def test_build_tts_blank_voice_falls_back_to_catalog_default(self):
        values = {
            "LINKDOG_TTS_BACKEND": "pocket",
            "LINKDOG_POCKET_VOICE": "   ",
        }
        with (
            patch.dict("os.environ", values, clear=True),
            patch.object(main, "_POCKET_TTS_BACKEND", None),
        ):
            backend = main.build_tts()

        self.assertEqual(backend.voice, "cosette")

    def test_build_tts_blank_backend_is_not_pocket(self):
        # Unset/blank backend resolves to 'command', never to pocket.
        values = {"LINKDOG_TTS_BACKEND": "   "}
        with (
            patch.dict("os.environ", values, clear=True),
            patch.object(main, "_POCKET_TTS_BACKEND", None),
        ):
            backend = main.build_tts()

        self.assertNotIsInstance(backend, PocketTTSBackend)

    def test_build_tts_reuses_singleton_for_same_normalised_voice(self):
        # 'COSETTE' and ' cosette ' normalise to the same voice, so the cached
        # backend must be reused rather than rebuilt.
        #
        # The `_POCKET_TTS_BACKEND` patch must wrap BOTH calls: leaving the
        # inner context restores the attribute to its previous value and would
        # clear the cache between calls, which would test nothing.
        with patch.object(main, "_POCKET_TTS_BACKEND", None):
            with patch.dict(
                "os.environ",
                {"LINKDOG_TTS_BACKEND": "pocket", "LINKDOG_POCKET_VOICE": "COSETTE"},
                clear=True,
            ):
                first = main.build_tts()

            with patch.dict(
                "os.environ",
                {"LINKDOG_TTS_BACKEND": "pocket", "LINKDOG_POCKET_VOICE": " cosette "},
                clear=True,
            ):
                second = main.build_tts()

        self.assertIsInstance(first, PocketTTSBackend)
        self.assertEqual(first.voice, "cosette")
        self.assertIs(first, second)

    def test_build_tts_rebuilds_when_normalised_voice_changes(self):
        with patch.object(main, "_POCKET_TTS_BACKEND", None):
            with patch.dict(
                "os.environ",
                {"LINKDOG_TTS_BACKEND": "pocket", "LINKDOG_POCKET_VOICE": "cosette"},
                clear=True,
            ):
                first = main.build_tts()

            with patch.dict(
                "os.environ",
                {"LINKDOG_TTS_BACKEND": "pocket", "LINKDOG_POCKET_VOICE": "alba"},
                clear=True,
            ):
                second = main.build_tts()

        self.assertEqual(second.voice, "alba")
        self.assertIsNot(first, second)


# --------------------------------------------------------------------------
# Defect 2: remote sources must not be existence-checked locally
# --------------------------------------------------------------------------


class RemoteSourcePreflightTests(unittest.TestCase):
    HF_STATE_URL = (
        "hf://kyutai/pocket-tts-without-voice-cloning/languages/english"
        "/embeddings/cosette.safetensors"
    )

    def test_remote_state_source_is_not_rejected(self):
        # Pre-fix: Path.exists() == False -> FATAL -> preflight aborts the
        # adapter, even though the reference is valid and simply remote.
        model = SimpleNamespace(has_voice_cloning=False)

        def get_state(voice):
            return {"voice": voice}

        def generate_audio_stream(state, text, copy_state=True):
            import numpy as np

            yield np.asarray([0.1, 0.2], dtype=np.float32)

        model.get_state_for_audio_prompt = get_state
        model.generate_audio_stream = generate_audio_stream

        env = {
            "LINKDOG_TTS_BACKEND": "pocket",
            "LINKDOG_POCKET_VOICE": self.HF_STATE_URL,
        }
        out = []
        code = preflight.run(model_factory=lambda: model, env=env, out=out)

        self.assertEqual(code, 0, f"unexpected abort: {out}")
        self.assertFalse(any("does not exist" in line for line in out))

    def test_is_remote_source_classification(self):
        self.assertTrue(preflight.is_remote_source(self.HF_STATE_URL))
        self.assertTrue(preflight.is_remote_source("https://example.com/v.safetensors"))
        self.assertTrue(preflight.is_remote_source("http://example.com/v.wav"))
        self.assertFalse(preflight.is_remote_source("/tmp/v.wav"))
        self.assertFalse(preflight.is_remote_source("voices/v.wav"))
        self.assertFalse(preflight.is_remote_source("cosette"))

    def test_local_missing_path_still_aborts(self):
        # The fix must not weaken the local case the check exists for.
        env = {
            "LINKDOG_TTS_BACKEND": "pocket",
            "LINKDOG_POCKET_VOICE": "/tmp/definitely-missing-voice-98765.wav",
        }
        out = []
        code = preflight.run(model_factory=lambda: None, env=env, out=out)

        self.assertNotEqual(code, 0)
        self.assertTrue(any("does not exist" in line for line in out))


# --------------------------------------------------------------------------
# Defect 3: root cause must survive truncation
# --------------------------------------------------------------------------


class FormatCauseTests(unittest.TestCase):
    def _build_chain(self, outer_len):
        root = ValueError("ROOT_CAUSE_SENTINEL voice cloning unavailable")
        outer = TTSError("x" * outer_len)
        outer.__cause__ = root
        return outer

    def test_root_cause_visible_with_short_outer_message(self):
        rendered = voice_turn.VoiceTurnWorker._format_cause(self._build_chain(20))
        self.assertIn("ROOT_CAUSE_SENTINEL", rendered)

    def test_root_cause_visible_with_long_outer_message(self):
        # Pre-fix: the 600-char head slice consumed the entire budget and the
        # root cause fell past the cut (measured: root visible == False).
        rendered = voice_turn.VoiceTurnWorker._format_cause(self._build_chain(800))
        self.assertIn("ROOT_CAUSE_SENTINEL", rendered)

    def test_root_cause_visible_with_very_long_outer_message(self):
        rendered = voice_turn.VoiceTurnWorker._format_cause(self._build_chain(8000))
        self.assertIn("ROOT_CAUSE_SENTINEL", rendered)

    def test_rendered_chain_stays_bounded(self):
        rendered = voice_turn.VoiceTurnWorker._format_cause(self._build_chain(50000))
        # Bounded well below the raw input; the exact ceiling is an
        # implementation detail but it must not grow with the outer message.
        self.assertLess(len(rendered), 2000)

    def test_no_infinite_loop_on_self_referential_chain(self):
        error = TTSError("loops")
        error.__cause__ = error
        rendered = voice_turn.VoiceTurnWorker._format_cause(error)
        self.assertIn("loops", rendered)


# --------------------------------------------------------------------------
# Defect 4: credential redaction on /health and in the failure log
# --------------------------------------------------------------------------


class RedactSecretsTests(unittest.TestCase):
    def test_url_token_parameter_is_masked(self):
        text = f"HTTPError 401 for url https://huggingface.co/api?token={SYNTHETIC_SECRET}"
        masked = redact_secrets(text)
        self.assertNotIn(SYNTHETIC_SECRET, masked)
        self.assertIn("token=", masked)
        self.assertIn("401", masked)  # diagnostic value preserved

    def test_authorization_bearer_value_is_masked(self):
        """Real leak found 2026-09-14: the Bearer value survived redaction.

        ``_KEYED_SECRET_RE`` matched ``Authorization`` and took the literal word
        ``Bearer`` as the value, leaving the actual credential intact. That
        substitution also deleted the word ``Bearer``, so ``_BEARER_RE`` could
        not match afterwards either — the two rules each relied on the other.

        The existing test used ``eyJhbG...ture`` as its sample, whose value
        fragment never contained the substring being asserted against, so the
        leak passed unnoticed. This test asserts on the *value* itself.
        """
        for scheme in ("Bearer", "bearer", "token"):
            with self.subTest(scheme=scheme):
                masked = redact_secrets(f"Authorization: {scheme} {SYNTHETIC_SECRET}")
                self.assertNotIn(SYNTHETIC_SECRET, masked)
                # The scheme survives so the line stays readable.
                self.assertIn(scheme, masked)
                self.assertIn("[REDACTED]", masked)

    def test_curl_style_quoted_bearer_is_masked(self):
        masked = redact_secrets(f"curl -H 'Authorization: Bearer {SYNTHETIC_SECRET}'")
        self.assertNotIn(SYNTHETIC_SECRET, masked)

    def test_bearer_redaction_is_idempotent(self):
        """Re-running must not re-mangle an already-redacted line."""
        once = redact_secrets(f"Authorization: Bearer {SYNTHETIC_SECRET}")
        self.assertEqual(redact_secrets(once), once)

    def test_various_credential_shapes_are_masked(self):
        cases = [
            "access_token=hf_ABCDEFGHIJKLMNOPQRSTUV",
            "api_key=topsecretvalue123",
            "password=hunter2hunter2",
            "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature",
            "curl -H 'Authorization: token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'",
            f"https://x/y?sig=deadbeefdeadbeef&token={SYNTHETIC_SECRET}",
        ]
        for case in cases:
            with self.subTest(case=case):
                masked = redact_secrets(case)
                self.assertNotIn(SYNTHETIC_SECRET, masked)
                for needle in (
                    "hf_ABCDEFGHIJKLMNOPQRSTUV",
                    "topsecretvalue123",
                    "hunter2hunter2",
                    "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    "deadbeefdeadbeef",
                ):
                    self.assertNotIn(needle, masked)

    def test_redaction_is_idempotent(self):
        text = f"token={SYNTHETIC_SECRET}"
        once = redact_secrets(text)
        self.assertEqual(redact_secrets(once), once)

    def test_round_two_leak_shapes_are_all_masked(self):
        """Astra round 2 (R2) measured 6 of these 8 shapes leaking.

        The prior fix only taught the key/regex pair about the ``bearer``,
        ``token`` and ``basic`` *schemes*, so every shape that put something
        between the key and the value (a quote, a different scheme, a nested
        ``=``, a path separator) walked straight through. This pins all eight so
        the next "helpful" scheme-by-scheme patch is caught immediately.
        """
        cases = [
            f'Authorization: Basic {SYNTHETIC_SECRET}',
            f'Authorization: Negotiate {SYNTHETIC_SECRET}',
            f'Authorization: Digest response="{SYNTHETIC_SECRET}"',
            f'token=abcd={SYNTHETIC_SECRET}',
            f'Bearer abcdefgh/{SYNTHETIC_SECRET}',
            f'token={SYNTHETIC_SECRET}',
            f'"Authorization": "Bearer {SYNTHETIC_SECRET}"',
            f'"X-Api-Key": "{SYNTHETIC_SECRET}"',
        ]
        for case in cases:
            with self.subTest(case=case):
                masked = redact_secrets(case)
                self.assertNotIn(SYNTHETIC_SECRET, masked)

    def test_round_two_leak_shapes_are_idempotent(self):
        """Masking must reach a fixed point, or repeated log passes re-mangle.

        Round 2 also flagged that the per-scheme patch broke idempotency: the
        first pass stripped the scheme word, so a second pass took the *scheme*
        as the value and redacted that too.
        """
        cases = [
            f'Authorization: Basic {SYNTHETIC_SECRET}',
            f'Authorization: Negotiate {SYNTHETIC_SECRET}',
            f'"Authorization": "Bearer {SYNTHETIC_SECRET}"',
            f'"X-Api-Key": "{SYNTHETIC_SECRET}"',
        ]
        for case in cases:
            with self.subTest(case=case):
                once = redact_secrets(case)
                self.assertEqual(redact_secrets(once), once)

    def test_round_three_leak_shapes_are_all_masked(self):
        """Astra round 3 (R2): four shapes still leaked after the round-2 fix.

        Each one is a *class* of leak, not a one-off string, so the fix was
        structural. These cases pin the structure, not the enumeration:
        an RFC 7235 scheme containing ``+``, an unknown scheme with a parameter
        list, a JSON value holding an escaped quote, and URL userinfo.
        """
        cases = {
            "rfc_scheme_with_plus": f"Authorization: Custom+Auth {SYNTHETIC_SECRET}",
            "unknown_scheme_param_list": (
                f'Authorization: Custom "{SYNTHETIC_SECRET}", second="b"'
            ),
            "escaped_quote_in_json": json.dumps(
                {"password": 'prefix\\"' + SYNTHETIC_SECRET}
            ),
            "url_userinfo": f"https://user:{SYNTHETIC_SECRET}@example.invalid/voice.wav",
        }
        for name, case in cases.items():
            with self.subTest(case=name):
                masked = redact_secrets(case)
                self.assertNotIn(SYNTHETIC_SECRET, masked)

    def test_round_three_shapes_are_idempotent(self):
        cases = [
            f"Authorization: Custom+Auth {SYNTHETIC_SECRET}",
            f'Authorization: Custom "{SYNTHETIC_SECRET}", second="b"',
            f"https://user:{SYNTHETIC_SECRET}@example.invalid/voice.wav",
        ]
        for case in cases:
            with self.subTest(case=case):
                once = redact_secrets(case)
                self.assertEqual(redact_secrets(once), once)

    def test_masked_json_stays_parseable(self):
        """A JSON body must survive redaction as valid JSON.

        Round 3: the regex consumed the closing quote of the value, so a masked
        ``{"Authorization": "Basic ..." }`` could no longer be ``json.loads``-ed.
        Log and health payloads are machine-read, so silently corrupting the
        structure is its own defect.
        """
        for doc in (
            {"Authorization": f"Basic {SYNTHETIC_SECRET}"},
            {"X-Api-Key": SYNTHETIC_SECRET},
            {"Authorization": f"Bearer {SYNTHETIC_SECRET}"},
        ):
            with self.subTest(doc=doc):
                masked = redact_secrets(json.dumps(doc))
                parsed = json.loads(masked)  # must not raise
                self.assertNotIn(SYNTHETIC_SECRET, masked)
                self.assertIn("[REDACTED]", json.dumps(parsed))

    def test_bare_scheme_word_is_left_alone(self):
        """A scheme word with no value is not a secret.

        Round 3 (minor): ``redact_secrets("Digest")`` returned ``""``, which
        contradicted the documented "leave unchanged when there are no
        parameters" behaviour and destroyed diagnostic text.
        """
        for word in ("Digest", "Bearer", "Basic"):
            with self.subTest(word=word):
                self.assertEqual(redact_secrets(word), word)

    def test_ordinary_diagnostics_are_untouched(self):
        text = "ValueError: voice cloning unavailable for /tmp/butterfly_excited.wav"
        self.assertEqual(redact_secrets(text), text)

    def test_empty_and_none_are_safe(self):
        self.assertEqual(redact_secrets(""), "")
        self.assertEqual(redact_secrets(None), "")


class HealthRedactionTests(unittest.TestCase):
    def setUp(self):
        main.ACTIVE_SESSIONS.clear()
        self.client = TestClient(main.app)

    def tearDown(self):
        main.ACTIVE_SESSIONS.clear()

    def test_health_never_exposes_secret_from_backend_last_error(self):
        # Pre-fix: last_error was published verbatim -> secret leaked = True.
        backend = SimpleNamespace(
            _model=None,
            _state=None,
            load_status="failed",
            last_error=(
                "HTTPError 401 for url "
                f"https://huggingface.co/api?token={SYNTHETIC_SECRET}"
            ),
        )
        values = {
            "LINKDOG_TTS_BACKEND": "pocket",
            "LINKDOG_POCKET_VOICE": "cosette",
        }
        with (
            patch.dict("os.environ", values, clear=True),
            patch.object(main, "_POCKET_TTS_BACKEND", backend),
        ):
            response = self.client.get("/api/health")

        body = response.text
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(SYNTHETIC_SECRET, body)

    def test_health_never_exposes_secret_from_voice_string(self):
        """R3 (Astra round 2): ``voice`` reached /health verbatim.

        The voice string is operator-supplied and may be a full URL carrying a
        query token — and /health is unauthenticated. The pre-fix code returned
        it raw, so the fix to ``app/redact.py`` alone did not remove this leak.
        """
        backend = SimpleNamespace(
            _model=None,
            _state=None,
            load_status="ready",
            last_error=None,
        )
        values = {
            "LINKDOG_TTS_BACKEND": "pocket",
            "LINKDOG_POCKET_VOICE": (
                f"https://example.invalid/voice.wav?token={SYNTHETIC_SECRET}"
            ),
        }
        with (
            patch.dict("os.environ", values, clear=True),
            patch.object(main, "_POCKET_TTS_BACKEND", backend),
        ):
            response = self.client.get("/api/health")

        body = response.text
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(SYNTHETIC_SECRET, body)

    def test_health_still_reports_voice_identity_after_redaction(self):
        """Redaction must not destroy the diagnostic value of the field.

        Operators read /health to learn *which* voice is configured. Masking the
        credential is required; blanking the whole field is not.
        """
        backend = SimpleNamespace(
            _model=None,
            _state=None,
            load_status="ready",
            last_error=None,
        )
        values = {
            "LINKDOG_TTS_BACKEND": "pocket",
            "LINKDOG_POCKET_VOICE": (
                f"https://example.invalid/cosette.wav?token={SYNTHETIC_SECRET}"
            ),
        }
        with (
            patch.dict("os.environ", values, clear=True),
            patch.object(main, "_POCKET_TTS_BACKEND", backend),
        ):
            snapshot = main._tts_health_snapshot()

        self.assertIn("cosette.wav", snapshot["voice"])
        self.assertNotIn(SYNTHETIC_SECRET, snapshot["voice"])

    def test_health_reports_credential_source(self):
        """R6 (Astra round 2): provenance was dropped before /health.

        ``token_source`` was computed and then discarded in
        ``PocketTTSBackend._diagnose_cloning``, so monitoring could not see that
        a working credential was actually cache-sourced and one cache clear away
        from breaking. Two distinct paths must carry it: the live provenance
        block, and the source the backend observed while diagnosing a failure.
        """
        from app import hf_token

        backend = SimpleNamespace(
            _model=None,
            _state=None,
            load_status="ready",
            last_error=None,
            cloning_credential_source=hf_token.SOURCE_HUB_CACHE,
            cloning_credential_durable=False,
        )
        values = {
            "LINKDOG_TTS_BACKEND": "pocket",
            "LINKDOG_POCKET_VOICE": "cosette",
        }
        fake = hf_token.CredentialProvenance(
            source=hf_token.SOURCE_HUB_CACHE,
            durable=False,
            degraded=True,
            hub_token_path="/tmp/synthetic/token",
            detail="synthetic",
        )
        with (
            patch.dict("os.environ", values, clear=True),
            patch.object(main, "_POCKET_TTS_BACKEND", backend),
            patch.object(main, "credential_provenance", lambda env=None: fake),
        ):
            snapshot = main._tts_health_snapshot()

        # The live view: what the Hub resolves right now.
        self.assertEqual(snapshot["credential"]["source"], hf_token.SOURCE_HUB_CACHE)
        self.assertFalse(snapshot["credential"]["durable"])
        self.assertTrue(snapshot["credential"]["degraded"])
        # The observed view: what the backend saw when it diagnosed a failure.
        self.assertEqual(snapshot["credential_source"], hf_token.SOURCE_HUB_CACHE)
        # R6: durability must be published too — a source alone still lets a
        # cache-sourced credential read as healthy. Identity, not truthiness:
        # `None` (never published) is falsy and would satisfy assertFalse.
        self.assertIs(snapshot["credential_durable"], False)
        # A degraded credential must be visible, never silently healthy.
        self.assertTrue(snapshot["credential"]["degraded"])

    def test_health_provenance_failure_does_not_break_the_endpoint(self):
        """R6 hardening: a broken probe must not take down an unauthenticated
        monitoring endpoint — but it must also not silently claim health."""
        backend = SimpleNamespace(
            _model=None,
            _state=None,
            load_status="ready",
            last_error=None,
        )
        values = {
            "LINKDOG_TTS_BACKEND": "pocket",
            "LINKDOG_POCKET_VOICE": "cosette",
        }

        def exploding_probe(env=None):
            raise RuntimeError("probe exploded")

        with (
            patch.dict("os.environ", values, clear=True),
            patch.object(main, "_POCKET_TTS_BACKEND", backend),
            patch.object(main, "credential_provenance", exploding_probe),
        ):
            snapshot = main._tts_health_snapshot()

        self.assertEqual(snapshot["credential"]["source"], "unknown")
        self.assertIsNone(snapshot["credential"]["durable"])

    def test_backend_credential_source_is_not_fabricated_before_a_probe(self):
        """``None`` until a failure actually diagnosed something."""
        backend = SimpleNamespace(
            _model=None,
            _state=None,
            load_status="ready",
            last_error=None,
        )
        self.assertIsNone(main.credential_source_from_backend(backend))
        self.assertIsNone(main.credential_source_from_backend(None))

    def test_pocket_backend_records_source_when_diagnosing_a_failure(self):
        """The R6 seam itself: ``_diagnose_cloning`` must not drop the source."""
        from app import hf_token
        from app.tts_auth import GatedAccessDiagnosis, OK

        # Must be the type the probe really returns: `token_source` and
        # `durable` live on `GatedAccessDiagnosis`, not on CredentialProvenance.
        diagnosis = GatedAccessDiagnosis(
            OK,
            detail="synthetic",
            token_source=hf_token.SOURCE_HUB_CACHE,
            durable=False,
        )
        # `_auth_probe` is the constructor-injected seam that `_diagnose_cloning`
        # calls; patching `app.pocket_tts.diagnose_gated_access` would not be
        # reached once a probe was injected.
        backend = PocketTTSBackend(
            voice="cloning:whatever", auth_probe=lambda *a, **k: diagnosis
        )
        backend._diagnose_cloning()

        self.assertEqual(
            backend.cloning_credential_source, hf_token.SOURCE_HUB_CACHE
        )
        # Identity, not truthiness: the un-recorded default is `None`, which is
        # falsy, so `assertFalse` would pass even if durability was never stored.
        # Mutant M6 (drop `durable`) survived against `assertFalse` — that was a
        # real gap, not a harness artifact.
        self.assertIs(backend.cloning_credential_durable, False)

    def test_pocket_backend_does_not_inherit_durability_across_observations(self):
        """R6 (Astra round 3): source and durability must come from ONE probe.

        Round 3 reproduced a mixed observation: a first diagnosis reporting
        ``project``/``durable=True`` followed by a second reporting
        ``hub_cache`` with no verdict left ``hub_cache`` + ``durable=True`` —
        a stale verdict published as if it described the current source. A
        ``None`` verdict means "not observed", never "still true".
        """
        from app import hf_token
        from app.tts_auth import GatedAccessDiagnosis, OK

        backend = PocketTTSBackend(voice="cloning:whatever")
        # First observation: durable project credential.
        backend._auth_probe = lambda *a, **k: GatedAccessDiagnosis(
            OK,
            detail="synthetic",
            token_source=hf_token.SOURCE_PROJECT,
            durable=True,
        )
        backend._diagnose_cloning()
        self.assertIs(backend.cloning_credential_durable, True)

        # Second observation: a different source, durability NOT determined.
        backend._auth_probe = lambda *a, **k: GatedAccessDiagnosis(
            OK,
            detail="synthetic",
            token_source=hf_token.SOURCE_HUB_CACHE,
            durable=None,
        )
        backend._diagnose_cloning()

        self.assertEqual(
            backend.cloning_credential_source, hf_token.SOURCE_HUB_CACHE
        )
        # Must be None ("not observed"), never the stale True.
        self.assertIsNone(backend.cloning_credential_durable)

    def test_pocket_backend_records_no_source_when_probe_is_inconclusive(self):
        """A probe with no source must not invent one, and must not crash."""
        backend = PocketTTSBackend(
            voice="cloning:whatever", auth_probe=lambda *a, **k: None
        )
        outcome, detail = backend._diagnose_cloning()

        self.assertIsNone(outcome)
        self.assertIsNone(detail)
        self.assertIsNone(backend.cloning_credential_source)

    def test_pocket_backend_survives_an_exploding_probe(self):
        """The failure path must not be broken by a broken diagnostic."""

        def exploding_probe(*a, **k):
            raise RuntimeError("probe exploded")

        backend = PocketTTSBackend(
            voice="cloning:whatever", auth_probe=exploding_probe
        )
        outcome, detail = backend._diagnose_cloning()

        self.assertIsNone(outcome)
        self.assertIn("RuntimeError", detail)
        self.assertIsNone(backend.cloning_credential_source)

    def test_pocket_tts_backend_redacts_at_the_source(self):
        # The writer itself must mask, so any OTHER consumer is covered too.
        def exploding_factory():
            raise RuntimeError(
                f"401 from https://huggingface.co/api?token={SYNTHETIC_SECRET}"
            )

        backend = PocketTTSBackend(voice="cosette", model_factory=exploding_factory)
        with self.assertRaises(RuntimeError):
            backend._ensure_loaded()

        self.assertEqual(backend.load_status, "failed")
        self.assertIsNotNone(backend.last_error)
        self.assertNotIn(SYNTHETIC_SECRET, backend.last_error)

    def test_failure_log_redacts_secret(self):
        captured = []

        class CaptureHandler(logging.Handler):
            def emit(self, record):
                captured.append(record.getMessage())

        handler = CaptureHandler()
        handler.setLevel(logging.WARNING)
        voice_turn.LOGGER.addHandler(handler)
        try:
            from app.device_session import DeviceSession

            class FakeWebSocket:
                async def send_text(self, _text):
                    pass

                async def send_bytes(self, _payload):
                    pass

            session = DeviceSession("TEST:DOG", FakeWebSocket())
            worker = voice_turn.VoiceTurnWorker(
                session, None, None, chat=None, tts=None, player=None
            )
            root = ValueError(
                f"clone failed: https://huggingface.co/api?token={SYNTHETIC_SECRET}"
            )
            outer = TTSError("synthesis failed")
            outer.__cause__ = root
            worker._note_tts_failure(outer, "hello", 1)
        finally:
            voice_turn.LOGGER.removeHandler(handler)

        joined = "\n".join(captured)
        # The chained root cause must still be rendered (diagnostic value),
        # while the embedded credential must not be.
        self.assertIn("clone failed", joined)
        self.assertNotIn(SYNTHETIC_SECRET, joined)


# --------------------------------------------------------------------------
# Defect 5: a failed load must not be masked by the catalog branch
# --------------------------------------------------------------------------


class CatalogFailedLoadTests(unittest.TestCase):
    def setUp(self):
        main.ACTIVE_SESSIONS.clear()
        self.client = TestClient(main.app)

    def tearDown(self):
        main.ACTIVE_SESSIONS.clear()

    def _health_with(self, voice, load_status):
        backend = SimpleNamespace(
            _model=None,
            _state=None,
            load_status=load_status,
            last_error="boom",
        )
        values = {
            "LINKDOG_TTS_BACKEND": "pocket",
            "LINKDOG_POCKET_VOICE": voice,
        }
        with (
            patch.dict("os.environ", values, clear=True),
            patch.object(main, "_POCKET_TTS_BACKEND", backend),
        ):
            return self.client.get("/api/health").json()["tts"]

    def test_catalog_voice_with_failed_load_is_not_configured_ok(self):
        # Pre-fix: configured_ok == True while model_status == 'failed'.
        tts = self._health_with("cosette", "failed")

        self.assertEqual(tts["voice_kind"], "catalog")
        self.assertEqual(tts["model_status"], "failed")
        self.assertIsNot(tts["configured_ok"], True)

    def test_catalog_voice_without_observation_still_reports_ok(self):
        # The catalog convenience must survive: no load attempted yet is not
        # a failure for a catalog voice (preserves the existing behaviour).
        tts = self._health_with("cosette", "idle")

        self.assertEqual(tts["voice_kind"], "catalog")
        self.assertEqual(tts["model_status"], "unknown")
        self.assertTrue(tts["configured_ok"])

    def test_catalog_voice_loading_reports_ok(self):
        tts = self._health_with("cosette", "loading")

        self.assertEqual(tts["model_status"], "loading")
        self.assertTrue(tts["configured_ok"])

    def test_non_catalog_failed_load_still_reports_false(self):
        tts = self._health_with("/tmp/some-voice.wav", "failed")

        self.assertEqual(tts["model_status"], "failed")
        self.assertIs(tts["configured_ok"], False)


if __name__ == "__main__":
    unittest.main()
