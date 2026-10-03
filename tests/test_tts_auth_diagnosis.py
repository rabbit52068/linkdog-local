"""Tests for gated-weight access diagnosis (the silent-degradation root cause).

Background
----------
``pocket_tts`` downloads the voice-cloning weights and, on failure, silently
falls back to the ungated weights while setting ``has_voice_cloning = False``
(``tts_model.py`` lines ~168-172 and ~229-235). The original error — no token,
expired token, unaccepted licence terms, or no network — is discarded, and the
operator only ever sees the generic ``VOICE_CLONING_UNSUPPORTED`` message.

:mod:`app.tts_auth` recovers that lost cause. These tests pin down:

1. each outcome is produced for the right underlying condition;
2. the diagnosis NEVER carries the token (it flows through the process, so a
   leak here would reach logs and the unauthenticated ``/health``);
3. a probe that cannot run is reported as ``unknown``, never as a failure —
   an inconclusive probe must not abort a working adapter;
4. the preflight and the resident backend actually surface the diagnosis.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.tts_auth import (  # noqa: E402
    GATED_REPO,
    INVALID_TOKEN,
    NETWORK,
    NO_ACCESS,
    NO_TOKEN,
    OK,
    UNKNOWN,
    GatedAccessDiagnosis,
    _classify_exception,
    diagnose_gated_access,
)

# Canary: if this ever appears in a diagnosis, redaction has been defeated.
SECRET = "hf_CANARY_DoNotLeak_0123456789abcdef"


def _fake_exception(name: str = "HTTPError", status: int | None = None) -> BaseException:
    """Build an exception that looks like a huggingface_hub failure."""

    class Response:  # noqa: D401 - test double
        status_code = status

    exc_cls = type(name, (Exception,), {})
    exc = exc_cls("synthetic failure")
    exc.response = Response()  # type: ignore[attr-defined]
    return exc


def _raising(exc: BaseException):
    def _fn(*args, **kwargs):
        raise exc

    return _fn


class ClassifyExceptionTests(unittest.TestCase):
    """``_classify_exception`` maps real HF error shapes onto outcomes."""

    def test_gated_repo_error_is_no_access(self):
        # The canonical "token valid but terms not accepted" error.
        self.assertEqual(_classify_exception(_fake_exception("GatedRepoError")).outcome, NO_ACCESS)

    def test_repository_not_found_is_no_access_not_network(self):
        # HF returns 404 for a gated repo the token cannot see. Treating this
        # as "network" or "repo typo" would send the operator down the wrong path.
        self.assertEqual(
            _classify_exception(_fake_exception("RepositoryNotFoundError")).outcome, NO_ACCESS
        )

    def test_401_is_invalid_token(self):
        self.assertEqual(_classify_exception(_fake_exception("HTTPError", 401)).outcome, INVALID_TOKEN)

    def test_403_is_no_access(self):
        self.assertEqual(_classify_exception(_fake_exception("HTTPError", 403)).outcome, NO_ACCESS)

    def test_connect_errors_are_network(self):
        for name in ("ConnectError", "ConnectTimeout", "ReadTimeout", "TimeoutException"):
            with self.subTest(name=name):
                self.assertEqual(_classify_exception(_fake_exception(name)).outcome, NETWORK)

    def test_connection_text_without_status_is_network(self):
        exc = Exception("Connection error: name resolution failed")
        self.assertEqual(_classify_exception(exc).outcome, NETWORK)

    def test_unrecognised_error_is_unknown_not_a_false_verdict(self):
        exc = Exception("something entirely unexpected")
        self.assertEqual(_classify_exception(exc).outcome, UNKNOWN)

    def test_classification_never_leaks_a_secret_in_detail(self):
        # A Hub error can embed the request URL, and the URL can carry ?token=.
        exc = Exception(f"401 Unauthorized for url https://huggingface.co/x?token={SECRET}")
        diagnosis = _classify_exception(exc)
        blob = diagnosis.describe() + str(diagnosis.detail)
        self.assertNotIn(SECRET, blob)


class DiagnoseGatedAccessTests(unittest.TestCase):
    """End-to-end behaviour of ``diagnose_gated_access`` with injected probes."""

    def test_no_token_is_reported_as_no_token(self):
        d = diagnose_gated_access(probes={"get_token": lambda: None})
        self.assertEqual(d.outcome, NO_TOKEN)
        self.assertFalse(d.ok)
        self.assertFalse(d.token_present)
        # The remedy must name the actual fix.
        self.assertIn("hf auth login", d.describe())

    def test_empty_string_token_is_no_token(self):
        d = diagnose_gated_access(probes={"get_token": lambda: ""})
        self.assertEqual(d.outcome, NO_TOKEN)

    def test_rejected_token_is_invalid_token(self):
        d = diagnose_gated_access(
            probes={
                "get_token": lambda: SECRET,
                "whoami": _raising(_fake_exception("HTTPError", 401)),
            }
        )
        self.assertEqual(d.outcome, INVALID_TOKEN)

    def test_valid_token_without_gated_access_is_no_access(self):
        d = diagnose_gated_access(
            probes={
                "get_token": lambda: SECRET,
                "whoami": lambda tok: None,
                "auth_check": _raising(_fake_exception("GatedRepoError")),
            }
        )
        self.assertEqual(d.outcome, NO_ACCESS)
        # The remedy must point at the licence acceptance page for the right repo.
        self.assertIn(GATED_REPO, d.describe())

    def test_fully_authorised_token_is_ok(self):
        d = diagnose_gated_access(
            probes={
                "get_token": lambda: SECRET,
                "whoami": lambda tok: None,
                "auth_check": lambda repo, tok: None,
            }
        )
        self.assertEqual(d.outcome, OK)

    def test_probe_that_cannot_run_is_unknown_not_failure(self):
        # An inconclusive probe must never look like a real problem: the
        # preflight turns non-ok outcomes into a hard abort.
        d = diagnose_gated_access(probes={"get_token": _raising(RuntimeError("boom"))})
        self.assertEqual(d.outcome, UNKNOWN)

    def test_network_failure_is_reported_as_network(self):
        d = diagnose_gated_access(
            probes={
                "get_token": lambda: SECRET,
                "whoami": _raising(_fake_exception("ConnectError")),
            }
        )
        self.assertEqual(d.outcome, NETWORK)

    def test_token_is_never_passed_to_a_diagnosis_string(self):
        # Exercise every failure branch and assert the canary never surfaces.
        branches = {
            "no_token": {"get_token": lambda: None},
            "invalid": {
                "get_token": lambda: SECRET,
                "whoami": _raising(_fake_exception("HTTPError", 401)),
            },
            "no_access": {
                "get_token": lambda: SECRET,
                "whoami": lambda tok: None,
                "auth_check": _raising(_fake_exception("GatedRepoError")),
            },
            "ok": {
                "get_token": lambda: SECRET,
                "whoami": lambda tok: None,
                "auth_check": lambda repo, tok: None,
            },
        }
        for label, probes in branches.items():
            with self.subTest(branch=label):
                d = diagnose_gated_access(probes=probes)
                blob = f"{d.outcome}|{d.detail}|{d.describe()}"
                self.assertNotIn(SECRET, blob)

    def test_exception_detail_carrying_a_token_is_redacted(self):
        exc = Exception(f"403 for url https://huggingface.co/api?token={SECRET}")
        d = diagnose_gated_access(
            probes={
                "get_token": lambda: SECRET,
                "whoami": lambda tok: None,
                "auth_check": _raising(exc),
            }
        )
        self.assertNotIn(SECRET, f"{d.detail}|{d.describe()}")


class GatedAccessDiagnosisObjectTests(unittest.TestCase):
    """The dataclass contract relied upon by preflight and /health."""

    def test_ok_flag_tracks_outcome(self):
        self.assertTrue(GatedAccessDiagnosis(OK).ok)
        self.assertFalse(GatedAccessDiagnosis(NO_ACCESS).ok)

    def test_describe_without_detail_has_no_empty_parens(self):
        line = GatedAccessDiagnosis(NO_TOKEN).describe()
        self.assertNotIn("()", line)
        self.assertNotIn("detail:", line)

    def test_describe_with_detail_includes_it(self):
        line = GatedAccessDiagnosis(UNKNOWN, "Weird: boom").describe()
        self.assertIn("Weird: boom", line)


class MetadataIsNotAuthorisationTests(unittest.TestCase):
    """The false-``ok`` that motivated switching the probe (second review).

    ``HfApi.model_info`` returns HTTP 200 for the gated repo with **no token at
    all**, because unauthenticated metadata is public. Measuring the live
    endpoint on 2026-09-14::

        GET /api/models/kyutai/pocket-tts            -> 200   (no auth header)
        GET /api/models/kyutai/pocket-tts/auth-check -> 401   (no auth header)

    So a successful metadata read cannot mean "authorised for the gated repo",
    and the earlier probe could report ``ok`` on a machine that had no usable
    credential. These tests lock in that the authorisation verdict is driven by
    the endpoint that actually enforces authorisation.
    """

    def test_metadata_only_success_is_not_treated_as_authorisation(self):
        """A probe whose metadata read succeeds but auth-check fails -> no_access."""
        d = diagnose_gated_access(
            probes={
                "get_token": lambda: SECRET,
                "whoami": lambda tok: None,
                # model_info would have returned 200 here; the authorisation
                # endpoint is what decides, and it rejects.
                "auth_check": _raising(_fake_exception("GatedRepoError")),
            }
        )
        self.assertEqual(d.outcome, NO_ACCESS)
        self.assertFalse(d.ok)

    def test_authorisation_endpoint_is_the_one_consulted(self):
        """auth_check must be called for the gated repo, exactly once."""
        calls: list[tuple[str, str]] = []

        def spy_auth_check(repo, tok):
            calls.append((repo, tok))

        d = diagnose_gated_access(
            probes={
                "get_token": lambda: SECRET,
                "whoami": lambda tok: None,
                "auth_check": spy_auth_check,
            }
        )

        self.assertEqual(d.outcome, OK)
        self.assertEqual(len(calls), 1, "authorisation must be checked exactly once")
        self.assertEqual(calls[0][0], GATED_REPO)

    def test_no_probe_is_skipped_when_metadata_would_have_passed(self):
        """Guard against the old probe silently returning ok with no auth check."""
        # Nothing supplied for auth_check means the real HfApi is imported; in a
        # network-free test we assert only that a provided seam is honoured.
        called = []

        d = diagnose_gated_access(
            probes={
                "get_token": lambda: SECRET,
                "whoami": lambda tok: None,
                "auth_check": lambda repo, tok: called.append(repo),
            }
        )
        self.assertEqual(called, [GATED_REPO])
        self.assertEqual(d.outcome, OK)

    def test_default_path_uses_auth_check_and_never_model_info(self):
        """The no-seam path must call auth_check — that is the whole fix.

        Every other test in this class injects an ``auth_check`` seam, so none
        of them exercise the *real* fallback. Reverting the production fallback
        to a metadata read (``model_info``) therefore left the suite green — a
        mutant survived on the first run of the reverse-verification harness
        (2026-09-14), which is how this gap was found.

        Here the real ``HfApi`` is replaced, and we assert that ``auth_check``
        is what gets called for the gated repo. ``model_info`` must not be
        consulted at all: it returns HTTP 200 anonymously, so using it as the
        authorisation probe produces a false ``ok`` with no usable credential.
        """
        calls = {"auth_check": [], "model_info": []}

        class FakeHfApi:
            def whoami(self, token=None):
                return {"name": "synthetic"}

            def auth_check(self, repo_id, token=None):
                calls["auth_check"].append(repo_id)

            def model_info(self, repo_id, token=None):
                calls["model_info"].append(repo_id)
                return {"id": repo_id}

        with patch("huggingface_hub.HfApi", FakeHfApi):
            d = diagnose_gated_access(probes={"get_token": lambda: SECRET})

        self.assertEqual(d.outcome, OK)
        self.assertEqual(
            calls["auth_check"],
            [GATED_REPO],
            "the real fallback must verify authorisation via auth_check",
        )
        self.assertEqual(
            calls["model_info"],
            [],
            "model_info returns 200 anonymously and proves nothing",
        )

    def test_default_path_reports_no_access_when_auth_check_rejects(self):
        """The real fallback must surface a rejection as NO_ACCESS, not ok.

        Third review (R5): this test used to assert only ``not OK``, and its
        double raised a plain ``RuntimeError`` whose *message* merely mentioned
        a 403. Classification reads the exception TYPE and ``status_code``, not
        message text, so a generic error correctly degrades to UNKNOWN — and
        ``not OK`` was satisfied by that UNKNOWN. The test therefore never
        pinned the rejection path it names. The double now raises the
        exception shape ``huggingface_hub`` actually raises, and the assertion
        names the outcome.
        """

        class GatedRepoError(Exception):
            """Same class name the Hub raises for an unauthorised gated repo."""

        class RejectingHfApi:
            def whoami(self, token=None):
                return {"name": "synthetic"}

            def auth_check(self, repo_id, token=None):
                raise GatedRepoError(f"403 Forbidden for {repo_id}")

            def model_info(self, repo_id, token=None):
                return {"id": repo_id}  # would have masked the failure

        with patch("huggingface_hub.HfApi", RejectingHfApi):
            d = diagnose_gated_access(probes={"get_token": lambda: SECRET})

        self.assertEqual(d.outcome, NO_ACCESS)
        self.assertFalse(d.ok)

    def test_auth_check_error_is_classified_not_merely_degraded(self):
        """Guard against re-weakening the test above into a `not OK` check.

        A rejection and an unexpected error must NOT collapse into the same
        outcome: ``NO_ACCESS`` says "the credential is not authorised" (an
        operator-fixable state) while ``UNKNOWN`` says "we could not tell".
        """

        class GatedRepoError(Exception):
            pass

        class RejectingHfApi:
            def whoami(self, token=None):
                return {"name": "synthetic"}

            def auth_check(self, repo_id, token=None):
                raise GatedRepoError("403 Forbidden")

        class BrokenHfApi:
            def whoami(self, token=None):
                return {"name": "synthetic"}

            def auth_check(self, repo_id, token=None):
                raise RuntimeError("something entirely unexpected")

        with patch("huggingface_hub.HfApi", RejectingHfApi):
            rejected = diagnose_gated_access(probes={"get_token": lambda: SECRET})
        with patch("huggingface_hub.HfApi", BrokenHfApi):
            broken = diagnose_gated_access(probes={"get_token": lambda: SECRET})

        self.assertEqual(rejected.outcome, NO_ACCESS)
        self.assertNotEqual(
            broken.outcome,
            NO_ACCESS,
            "an unrecognised error must not be reported as a permission verdict",
        )


class CredentialSourceReportingTests(unittest.TestCase):
    """A working-but-cache-sourced credential must not look like a healthy fix."""

    def test_token_source_is_recorded_and_never_alters_the_outcome(self):
        d = diagnose_gated_access(
            probes={
                "get_token": lambda: SECRET,
                "whoami": lambda tok: None,
                "auth_check": lambda repo, tok: None,
            },
            token_source="hub_cache",
        )
        self.assertEqual(d.outcome, OK)
        self.assertEqual(d.token_source, "hub_cache")

    def test_token_source_appears_in_describe(self):
        d = diagnose_gated_access(
            probes={"get_token": lambda: None},
            token_source="project",
        )
        self.assertIn("token source: project", d.describe())

    def test_token_source_is_auto_detected_when_not_supplied(self):
        """A forgetful caller must not be able to hide a cache-sourced token."""
        from app import hf_token

        # Third review: this used to fake only `constants` and assert on the
        # *path*, which is exactly the comparison that produced false positives.
        # Provenance now asks the Hub what it resolved, so the seam to inject is
        # `resolve_hub_token`.
        with (
            patch.object(hf_token, "resolve_hub_token", lambda env=None: SECRET),
            patch.object(hf_token, "_hub_constants", lambda: None),
        ):
            d = diagnose_gated_access(
                probes={
                    "get_token": lambda: SECRET,
                    "whoami": lambda tok: None,
                    "auth_check": lambda repo, tok: None,
                }
            )

        # No Hub constants and no matching project file: the source cannot be
        # attributed to the project, so it must NOT be reported as `project`.
        self.assertNotEqual(d.token_source, hf_token.SOURCE_PROJECT)
        self.assertEqual(d.outcome, OK)

    def test_token_source_never_carries_the_secret(self):
        d = diagnose_gated_access(
            probes={"get_token": lambda: SECRET},
            token_source="project",
        )
        self.assertNotIn(SECRET, d.describe() + d.token_source)


class BackendSurfacesDiagnosisTests(unittest.TestCase):
    """PocketTTSBackend records the cause when a cloning load fails."""

    def _backend(self, probe):
        from app.pocket_tts import PocketTTSBackend

        def failing_factory():
            raise RuntimeError("VOICE_CLONING_UNSUPPORTED style failure")

        return PocketTTSBackend(
            voice="/tmp/some_voice.wav",  # audio kind -> requires cloning
            model_factory=failing_factory,
            auth_probe=probe,
        )

    def test_failed_cloning_load_records_the_diagnosis(self):
        backend = self._backend(lambda: GatedAccessDiagnosis(NO_TOKEN))
        with self.assertRaises(RuntimeError):
            backend._ensure_loaded()
        self.assertEqual(backend.load_status, "failed")
        self.assertEqual(backend.cloning_diagnosis, NO_TOKEN)

    def test_catalog_voice_failure_does_not_probe(self):
        # A catalog voice never needs cloning, so probing would be a pointless
        # network call. The probe must not run.
        from app.pocket_tts import PocketTTSBackend

        calls = []

        def probe():
            calls.append(1)
            return GatedAccessDiagnosis(OK)

        def failing_factory():
            raise RuntimeError("boom")

        backend = PocketTTSBackend(
            voice="cosette", model_factory=failing_factory, auth_probe=probe
        )
        with self.assertRaises(RuntimeError):
            backend._ensure_loaded()
        self.assertEqual(calls, [], "probe must not run for a catalog voice")
        self.assertIsNone(backend.cloning_diagnosis)

    def test_a_broken_probe_never_breaks_the_failure_path(self):
        # The diagnostic is a nicety; it must not mask or replace the real error.
        backend = self._backend(_raising(RuntimeError("probe exploded")))
        with self.assertRaises(RuntimeError) as ctx:
            backend._ensure_loaded()
        self.assertIn("VOICE_CLONING_UNSUPPORTED style failure", str(ctx.exception))
        self.assertEqual(backend.load_status, "failed")
        self.assertIsNone(backend.cloning_diagnosis)

    def test_probe_returning_none_is_tolerated(self):
        backend = self._backend(lambda: None)
        with self.assertRaises(RuntimeError):
            backend._ensure_loaded()
        self.assertIsNone(backend.cloning_diagnosis)


class PreflightSurfacesDiagnosisTests(unittest.TestCase):
    """The preflight prints the precise cause instead of guessing."""

    def setUp(self):
        # The existence check runs *before* the cloning branch, so the voice
        # must be a real file or the preflight aborts for an unrelated reason.
        import tempfile

        self._tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self._tmp.write(b"\x00" * 1024)
        self._tmp.close()
        self.voice_path = self._tmp.name

    def tearDown(self):
        Path(self.voice_path).unlink(missing_ok=True)

    def _run_with_no_cloning_model(self, probe):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "tts_preflight_mod", REPO_ROOT / "scripts" / "tts_preflight.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class FakeModel:
            has_voice_cloning = False

            def get_state_for_audio_prompt(self, voice):
                raise ValueError("should not be reached")

        lines: list[str] = []
        code = module.run(
            model_factory=lambda: FakeModel(),
            env={
                "LINKDOG_TTS_BACKEND": "pocket",
                "LINKDOG_POCKET_VOICE": self.voice_path,
            },
            out=lines,
            auth_probe=probe,
        )
        return code, "\n".join(lines)

    def test_preflight_reports_no_token_precisely(self):
        code, text = self._run_with_no_cloning_model(lambda: GatedAccessDiagnosis(NO_TOKEN))
        self.assertEqual(code, 1)
        self.assertIn("no_token", text)
        self.assertIn("hf auth login", text)

    def test_preflight_mentions_the_silent_fallback(self):
        code, text = self._run_with_no_cloning_model(lambda: GatedAccessDiagnosis(NO_ACCESS))
        self.assertEqual(code, 1)
        self.assertIn("UNGATED", text.upper())
        self.assertIn(GATED_REPO, text)

    def test_preflight_survives_a_broken_probe(self):
        # Must still abort (the voice genuinely cannot work) but must not crash.
        code, text = self._run_with_no_cloning_model(_raising(RuntimeError("probe died")))
        self.assertEqual(code, 1)
        self.assertIn("diagnosis itself failed", text)

    def test_preflight_never_prints_a_token(self):
        exc = Exception(f"403 url=https://hf.co/api?token={SECRET}")
        code, text = self._run_with_no_cloning_model(
            lambda: diagnose_gated_access(
                probes={
                    "get_token": lambda: SECRET,
                    "whoami": lambda tok: None,
                    "auth_check": _raising(exc),
                }
            )
        )
        self.assertEqual(code, 1)
        self.assertNotIn(SECRET, text)


class HealthExposesDiagnosisTests(unittest.TestCase):
    """/health carries the diagnosis so monitoring sees the reason, not just False."""

    def test_health_reports_diagnosis_fields_for_pocket(self):
        from unittest import mock

        import app.main as main

        class FakeBackend:
            load_status = "failed"
            _model = None
            _state = None
            last_error = "ValueError: VOICE_CLONING_UNSUPPORTED"
            cloning_diagnosis = NO_TOKEN
            cloning_diagnosis_detail = None

        env = {
            "LINKDOG_TTS_BACKEND": "pocket",
            "LINKDOG_POCKET_VOICE": "/tmp/voice.wav",
        }
        with mock.patch.dict("os.environ", env, clear=False), mock.patch.object(
            main, "_POCKET_TTS_BACKEND", FakeBackend()
        ):
            snapshot = main._tts_health_snapshot()

        self.assertEqual(snapshot["cloning_diagnosis"], NO_TOKEN)
        self.assertEqual(snapshot["model_status"], "failed")
        # A failed load must never read as configured_ok.
        self.assertFalse(snapshot["configured_ok"])

    def test_non_pocket_backend_reports_null_diagnosis(self):
        from unittest import mock

        import app.main as main

        with mock.patch.dict("os.environ", {"LINKDOG_TTS_BACKEND": "edge"}, clear=False):
            snapshot = main._tts_health_snapshot()

        self.assertIsNone(snapshot["cloning_diagnosis"])
        self.assertIsNone(snapshot["cloning_diagnosis_detail"])


if __name__ == "__main__":
    unittest.main()
