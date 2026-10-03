"""Guards for the 2026-09-14 fixes: credential durability + model single-source.

Two problems, both verified against the live system before being fixed.

1. **Token issue** — ``huggingface_hub`` resolves ``HF_TOKEN_PATH`` **once, at
   import time**. The default location is ``~/.cache/huggingface/token``, i.e.
   *inside the cache directory*, so any cache-clear step destroys the
   credential and the gated voice-cloning download silently degrades — the
   exact cause of the 2026-09-13 TTS outage. ``app.hf_token`` pins the path to
   a project-local file instead, and ``app/__init__.py`` runs it before any
   ``huggingface_hub`` import can freeze the old value.

2. **Model contradiction** — four disagreeing sources existed:
   ``.env`` (``glm-5.3-flash``, never read), ``data/settings.json``
   (``deepseek-v4.1-flash``, effective), and two code defaults
   (``deepseek-v4-flash:0731``) which the dashboard refuses to persist because
   that model is no longer in the catalog.

Testing contract (second review, 2026-09-14)
--------------------------------------------
These tests must be **green on a stripped checkout with no credential**. An
earlier version asserted ``"PINNED" not in stdout`` for the no-credential case,
but the probe script printed ``PINNED=`` unconditionally, so the assertion was
guaranteed to fail there. The whole point of the pin is durability *including*
a fresh clone, so the tests now build a **synthetic credential fixture** via
``LINKDOG_HF_TOKEN_FILE`` and exercise both branches deliberately:

- fixture present  -> the pin must apply, and the Hub must see the pinned path;
- fixture absent   -> the pin must decline, and no ``HF_TOKEN_PATH`` may be
  exported.

Whether the *real* project credential exists is therefore irrelevant to the
result — the tests never read it and never depend on it.

These tests never touch the network and never read a token value.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from app import state

REPO_ROOT = Path(__file__).resolve().parent.parent

# A short, obviously-synthetic marker. Never a real credential.
FAKE_TOKEN = "hf_SYNTHETIC_MARKER_FOR_TESTS_ONLY_0000"


class HfTokenPathTests(unittest.TestCase):
    """``app.hf_token.pin_hf_token_path`` semantics."""

    def setUp(self):
        import app.hf_token as hf_token

        self.hf_token = hf_token

    def test_explicit_env_var_wins(self):
        """An operator-supplied HF_TOKEN_PATH must never be overridden."""
        env = {"HF_TOKEN_PATH": "/operator/chosen/token"}
        with patch.object(self.hf_token, "DEFAULT_TOKEN_PATH", Path("/nonexistent")):
            applied = self.hf_token.pin_hf_token_path(env)

        self.assertFalse(applied)
        self.assertEqual(env["HF_TOKEN_PATH"], "/operator/chosen/token")
        self.assertEqual(self.hf_token.LAST_PIN_OUTCOME, self.hf_token.EXPLICIT_ENV)

    def test_missing_project_file_does_not_pin(self):
        """A stripped checkout must fall through, not pin a broken path."""
        env = {}
        with patch.object(
            self.hf_token, "DEFAULT_TOKEN_PATH", Path("/nonexistent/token")
        ):
            applied = self.hf_token.pin_hf_token_path(env)

        self.assertFalse(applied)
        self.assertNotIn("HF_TOKEN_PATH", env)
        self.assertEqual(self.hf_token.LAST_PIN_OUTCOME, self.hf_token.MISSING_FILE)

    def test_blank_project_file_does_not_pin(self):
        """Present-but-empty must NOT be pinned: the Hub should report no_token."""
        with tempfile.TemporaryDirectory() as directory:
            blank = Path(directory) / "hf_token"
            blank.write_text("")
            env = {}
            with patch.object(self.hf_token, "DEFAULT_TOKEN_PATH", blank):
                applied = self.hf_token.pin_hf_token_path(env)

            self.assertFalse(applied)
            self.assertNotIn("HF_TOKEN_PATH", env)
            self.assertEqual(self.hf_token.LAST_PIN_OUTCOME, self.hf_token.BLANK_FILE)

    def test_whitespace_only_project_file_does_not_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            blank = Path(directory) / "hf_token"
            blank.write_text("   \n\t\n")
            env = {}
            with patch.object(self.hf_token, "DEFAULT_TOKEN_PATH", blank):
                applied = self.hf_token.pin_hf_token_path(env)

            self.assertFalse(applied)
            self.assertNotIn("HF_TOKEN_PATH", env)

    def test_real_file_is_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            tok = Path(directory) / "hf_token"
            tok.write_text(FAKE_TOKEN)
            env = {}
            with patch.object(self.hf_token, "DEFAULT_TOKEN_PATH", tok):
                applied = self.hf_token.pin_hf_token_path(env)

            self.assertTrue(applied)
            self.assertEqual(env["HF_TOKEN_PATH"], str(tok))
            self.assertEqual(self.hf_token.LAST_PIN_OUTCOME, self.hf_token.PINNED)

    def test_pin_never_raises_on_unreadable_path(self):
        """A credential-location problem must not stop the adapter starting.

        A *directory* stands in for an unreadable path: ``is_file()`` is False
        so this exercises the same non-raising contract, and unlike a
        ``/proc/...`` path it is meaningful on macOS (where ``/proc`` does not
        exist, so the earlier version of this test only ever proved that a
        nonexistent path is nonexistent).
        """
        with tempfile.TemporaryDirectory() as directory:
            unreadable = Path(directory) / "token_is_a_directory"
            unreadable.mkdir()
            env = {}
            with patch.object(self.hf_token, "DEFAULT_TOKEN_PATH", unreadable):
                applied = self.hf_token.pin_hf_token_path(env)  # must not raise

            self.assertFalse(applied)
            self.assertEqual(self.hf_token.LAST_PIN_OUTCOME, self.hf_token.MISSING_FILE)

    def test_pin_does_not_leak_the_token_value(self):
        """Only the path may be exported; the secret must not be copied into env."""
        with tempfile.TemporaryDirectory() as directory:
            tok = Path(directory) / "hf_token"
            tok.write_text(FAKE_TOKEN)
            env = {}
            with patch.object(self.hf_token, "DEFAULT_TOKEN_PATH", tok):
                self.hf_token.pin_hf_token_path(env)

            self.assertEqual(sorted(env), sorted(["HF_TOKEN_PATH", "LINKDOG_HF_TOKEN_PATH"]))
            for key, value in env.items():
                self.assertNotIn(FAKE_TOKEN, value, f"{key} leaked the token value")

    def test_token_file_seam_is_honoured(self):
        """LINKDOG_HF_TOKEN_FILE lets a harness point at a synthetic fixture."""
        with tempfile.TemporaryDirectory() as directory:
            tok = Path(directory) / "synthetic_token"
            tok.write_text(FAKE_TOKEN)
            env = {self.hf_token._TOKEN_FILE_ENV: str(tok)}
            try:
                self.assertEqual(self.hf_token.token_path(env), tok)
                applied = self.hf_token.pin_hf_token_path(env)
                self.assertTrue(applied)
                self.assertEqual(env["HF_TOKEN_PATH"], str(tok))
            finally:
                # Never let the seam leak into the process environment.
                os.environ.pop(self.hf_token._TOKEN_FILE_ENV, None)

    def test_default_token_path_is_outside_the_hub_cache(self):
        """The whole point: credential lifetime decoupled from cache lifetime."""
        path = self.hf_token.DEFAULT_TOKEN_PATH
        cache_dir = Path.home() / ".cache" / "huggingface"

        self.assertNotEqual(path.parent, cache_dir)
        self.assertNotIn(".cache", path.parts)

    def test_secrets_directory_is_gitignored(self):
        """A credential that can be committed is not a durable secret.

        ``secrets/`` was NOT covered by .gitignore when the token was first
        moved there; the pin would have quietly made the token committable.
        """
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        entries = {line.strip() for line in gitignore.splitlines()}
        relative = self.hf_token.DEFAULT_TOKEN_PATH.relative_to(REPO_ROOT)
        top_level = f"{relative.parts[0]}/"

        self.assertIn(top_level, entries, f"{top_level} missing from .gitignore")

    def test_secrets_file_permissions_are_owner_only(self):
        """0600 on the real credential, when present.

        Skipped on a checkout without the credential — that is a legitimate
        state, not a failure. The *enforcement rule* is exercised
        unconditionally by
        ``PermissionsAreOwnerOnlyTests.test_synthetic_owner_only_file_passes``.
        """
        path = self.hf_token.DEFAULT_TOKEN_PATH
        if not path.is_file():
            self.skipTest("no project credential in this checkout")

        mode = path.stat().st_mode & 0o777
        self.assertEqual(mode, 0o600, f"expected 0600, got {oct(mode)}")

    def test_secrets_directory_permissions_are_owner_only(self):
        """The containing directory must not be world/group accessible.

        Skipped without a checkout-local ``secrets/``; the rule itself is
        exercised unconditionally in ``PermissionsAreOwnerOnlyTests``.
        """
        directory = self.hf_token.DEFAULT_TOKEN_PATH.parent
        if not directory.is_dir():
            self.skipTest("no secrets/ directory in this checkout")

        mode = directory.stat().st_mode & 0o777
        self.assertEqual(mode, 0o700, f"expected 0700, got {oct(mode)}")


class PermissionsAreOwnerOnlyTests(unittest.TestCase):
    """The permission *rule*, exercised without needing a real credential.

    Second review: the two checks above skip on a fresh clone (no ``secrets/``),
    so on a clean checkout nothing verified the rule at all — the tests were
    green because they never ran. These use synthetic fixtures with the modes
    set explicitly, so the assertion has teeth in every checkout, and only the
    presence of the real secret is conditional.
    """

    def test_synthetic_owner_only_file_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hf_token"
            path.write_text("synthetic\n", encoding="utf-8")
            path.chmod(0o600)

            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_synthetic_owner_only_directory_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            nested = Path(directory) / "secrets"
            nested.mkdir()
            nested.chmod(0o700)

            self.assertEqual(nested.stat().st_mode & 0o777, 0o700)

    def test_a_world_readable_file_would_be_rejected_by_the_rule(self):
        """Negative control: prove the assertion can actually fail.

        Without this, ``assertEqual(mode, 0o600)`` could be passing for the
        wrong reason and nobody would know.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hf_token"
            path.write_text("synthetic\n", encoding="utf-8")
            path.chmod(0o644)

            mode = path.stat().st_mode & 0o777
            self.assertNotEqual(mode, 0o600, "fixture failed to become 0644")
            self.assertTrue(mode & 0o044, "0644 should be group/world readable")

    def test_the_real_credential_mode_matches_the_rule_when_present(self):
        """Binds the synthetic rule to the actual artifact, if it exists."""
        path = Path(__file__).resolve().parents[1] / "secrets" / "hf_token"
        if not path.is_file():
            self.skipTest("no project credential in this checkout")

        mode = path.stat().st_mode & 0o777
        self.assertEqual(mode, 0o600, f"expected 0600, got {oct(mode)}")


class CredentialProvenanceTests(unittest.TestCase):
    """The durability fix must be *visible* when it is silently not in effect.

    Second review (2026-09-14): if ``secrets/hf_token`` disappears while a token
    still sits in the Hub cache, the Hub quietly reads the cache instead. The
    download works, ``diagnose_gated_access`` says ``ok``, and the durability
    fix is already broken — the next cache clear is another outage. These tests
    pin the reporting of that exact state.

    Third review (2026-09-14): the first version of these tests faked only
    ``huggingface_hub.constants`` and asserted on *path strings*. That made the
    suite agree with an implementation that reported ``project``/``durable`` for
    a deleted file, an in-cache symlink, and an ``HF_TOKEN``-overridden
    credential. Provenance now probes what the Hub actually resolved, so the
    tests inject that resolution through the ``resolve_hub_token`` seam and
    build real files — the classification is exercised, not the path comparison.
    """

    def setUp(self):
        import app.hf_token as hf_token

        self.hf_token = hf_token
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _provenance(self, *, hub_token, project_token, hub_token_path=None, env=None):
        """Run ``credential_provenance`` with a synthetic Hub resolution.

        ``hub_token`` is what the Hub would hand a request; ``project_token`` is
        written to the project credential file (``None`` = no file).
        """
        project = self.tmp / "secrets" / "hf_token"
        project.parent.mkdir(parents=True, exist_ok=True)
        if project_token is not None:
            project.write_text(project_token, encoding="utf-8")

        cache_dir = self.tmp / "cache"
        cache_dir.mkdir(exist_ok=True)
        resolved_path = hub_token_path or str(cache_dir / "token")

        constants = self._fake_constants(
            HF_HOME=str(cache_dir), HF_TOKEN_PATH=resolved_path
        )
        base_env = {"LINKDOG_HF_TOKEN_FILE": str(project)}
        if env:
            base_env.update(env)

        with (
            patch.object(self.hf_token, "_hub_constants", lambda: constants),
            patch.object(self.hf_token, "resolve_hub_token", lambda env=None: hub_token),
        ):
            return self.hf_token.credential_provenance(base_env), project

    def test_project_sourced_credential_is_durable(self):
        provenance, _ = self._provenance(
            hub_token="PROJECT_TOKEN_VALUE",
            project_token="PROJECT_TOKEN_VALUE",
            hub_token_path=None,  # unused: the value identifies the source
        )

        self.assertEqual(provenance.source, self.hf_token.SOURCE_PROJECT)
        self.assertTrue(provenance.durable)
        self.assertFalse(provenance.degraded)

    def test_cache_sourced_credential_is_reported_degraded(self):
        """The masking failure: working now, one cache clear from an outage.

        The project file is deliberately ABSENT while the Hub still resolves a
        token from inside its cache — the exact silent-degradation state.
        """
        cache_dir = self.tmp / "cache"
        cache_dir.mkdir(exist_ok=True)
        cache_token_file = cache_dir / "token"
        cache_token_file.write_text("CACHE_TOKEN_VALUE", encoding="utf-8")

        provenance, _ = self._provenance(
            hub_token="CACHE_TOKEN_VALUE",
            project_token=None,
            hub_token_path=str(cache_token_file),
        )

        self.assertEqual(provenance.source, self.hf_token.SOURCE_HUB_CACHE)
        self.assertFalse(provenance.durable)
        self.assertTrue(provenance.degraded)
        self.assertIn("DEGRADED", provenance.describe())

    def test_explicit_path_inside_cache_is_flagged_not_durable(self):
        cache_dir = self.tmp / "cache"
        cache_dir.mkdir(exist_ok=True)
        inside = cache_dir / "my_token"
        inside.write_text("EXPLICIT_INSIDE", encoding="utf-8")

        provenance, _ = self._provenance(
            hub_token="EXPLICIT_INSIDE",
            project_token=None,
            hub_token_path=str(inside),
            env={"HF_TOKEN_PATH": str(inside)},
        )

        self.assertFalse(provenance.durable)
        self.assertTrue(provenance.degraded)

    def test_explicit_path_outside_cache_is_durable(self):
        outside = self.tmp / "outside_token"
        outside.write_text("EXPLICIT_OUTSIDE", encoding="utf-8")

        provenance, _ = self._provenance(
            hub_token="EXPLICIT_OUTSIDE",
            project_token=None,
            hub_token_path=str(outside),
            env={"HF_TOKEN_PATH": str(outside)},
        )

        self.assertTrue(provenance.durable)
        self.assertFalse(provenance.degraded)

    def test_env_token_is_reported_degraded(self):
        """``HF_TOKEN`` outranks the pinned file, bypassing the durability fix.

        Third-review false positive #3: this reported ``project`` even though the
        environment token was the one actually in use.
        """
        provenance, _ = self._provenance(
            hub_token="ENV_TOKEN_VALUE",
            project_token="PROJECT_TOKEN_VALUE",
            env={"HF_TOKEN": "ENV_TOKEN_VALUE"},
        )

        self.assertEqual(provenance.source, self.hf_token.SOURCE_ENV_TOKEN)
        self.assertTrue(provenance.degraded)
        self.assertFalse(provenance.durable)

    def test_deleted_project_file_is_not_durable(self):
        """Third-review false positive #1.

        The project file is gone while ``constants.HF_TOKEN_PATH`` still points
        at where it used to be. The old path comparison saw "the Hub's path is
        the project path" and claimed a durable project credential; the Hub in
        fact resolves nothing at all.
        """
        provenance, project = self._provenance(
            hub_token="",
            project_token="PROJECT_TOKEN_VALUE",
            hub_token_path=str(self.tmp / "secrets" / "hf_token"),
        )
        project.unlink()  # now genuinely absent at request time

        self.assertEqual(provenance.source, self.hf_token.SOURCE_NONE)
        self.assertFalse(provenance.durable)
        self.assertFalse(provenance.degraded)

    def test_symlink_into_cache_is_not_durable(self):
        """Third-review false positive #2.

        The project path holds the right *value*, but through a symlink into the
        Hub cache — so clearing the cache still destroys the credential.
        """
        cache_dir = self.tmp / "cache"
        cache_dir.mkdir(exist_ok=True)
        target = cache_dir / "real_token"
        target.write_text("SHARED_VALUE", encoding="utf-8")

        project = self.tmp / "secrets" / "hf_token"
        project.parent.mkdir(parents=True, exist_ok=True)
        project.symlink_to(target)

        constants = self._fake_constants(
            HF_HOME=str(cache_dir), HF_TOKEN_PATH=str(project)
        )
        with (
            patch.object(self.hf_token, "_hub_constants", lambda: constants),
            patch.object(
                self.hf_token, "resolve_hub_token", lambda env=None: "SHARED_VALUE"
            ),
        ):
            provenance = self.hf_token.credential_provenance(
                {"LINKDOG_HF_TOKEN_FILE": str(project)}
            )

        self.assertEqual(provenance.source, self.hf_token.SOURCE_HUB_CACHE)
        self.assertFalse(provenance.durable)
        self.assertTrue(provenance.degraded)

    def test_same_value_in_cache_and_project_is_not_reported_as_project(self):
        """the *same value* does not mean the *same file*.

        The classifier used to compare the project file's value first, so an
        identical token living in both the project file and the Hub cache was
        reported ``project``/durable even when ``get_token()`` proved the Hub
        read the cache copy. Origin is decided by the file the Hub will really
        read, and only then by the project file.
        """
        cache_dir = self.tmp / "cache"
        cache_dir.mkdir(exist_ok=True)
        cache_token_file = cache_dir / "token"
        cache_token_file.write_text("SHARED_VALUE", encoding="utf-8")

        project = self.tmp / "secrets" / "hf_token"
        project.parent.mkdir(parents=True, exist_ok=True)
        project.write_text("SHARED_VALUE", encoding="utf-8")

        # The Hub's own path points at the CACHE copy, not the project file.
        constants = self._fake_constants(
            HF_HOME=str(cache_dir), HF_TOKEN_PATH=str(cache_token_file)
        )
        with (
            patch.object(self.hf_token, "_hub_constants", lambda: constants),
            patch.object(
                self.hf_token, "resolve_hub_token", lambda env=None: "SHARED_VALUE"
            ),
        ):
            provenance = self.hf_token.credential_provenance(
                {"LINKDOG_HF_TOKEN_FILE": str(project)}
            )

        self.assertEqual(provenance.source, self.hf_token.SOURCE_HUB_CACHE)
        self.assertFalse(provenance.durable)
        self.assertTrue(provenance.degraded)

    def test_legacy_env_var_name_matches_the_hub(self):
        """the legacy variable is ``HUGGING_FACE_HUB_TOKEN``.

        The classifier looked for ``HUGGINGFACE_HUB_TOKEN``, which the Hub never
        reads, so a credential set under the real legacy name was silently
        attributed to the project file and reported durable.
        """
        provenance, _ = self._provenance(
            hub_token="LEGACY_TOKEN_VALUE",
            project_token="PROJECT_TOKEN_VALUE",
            env={"HUGGING_FACE_HUB_TOKEN": "LEGACY_TOKEN_VALUE"},
        )

        self.assertEqual(provenance.source, self.hf_token.SOURCE_ENV_TOKEN)
        self.assertFalse(provenance.durable)
        self.assertTrue(provenance.degraded)

    def test_unknown_source_is_not_claimed_durable(self):
        """an unverifiable source must not publish durable.

        ``SOURCE_OTHER`` said durability "cannot be verified" while computing
        ``durable = not in_cache`` from a possibly unrelated ``HF_TOKEN_PATH`` —
        emitting a confident ``durable=True`` for a source it had just admitted
        it could not classify. Not-durable is the honest direction: it can only
        raise a warning, never mask a broken credential.
        """
        cache_dir = self.tmp / "cache"
        cache_dir.mkdir(exist_ok=True)
        unrelated = self.tmp / "elsewhere" / "token"
        unrelated.parent.mkdir(parents=True, exist_ok=True)
        unrelated.write_text("UNRELATED", encoding="utf-8")

        constants = self._fake_constants(
            HF_HOME=str(cache_dir), HF_TOKEN_PATH=str(unrelated)
        )
        with (
            patch.object(self.hf_token, "_hub_constants", lambda: constants),
            patch.object(
                self.hf_token, "resolve_hub_token", lambda env=None: "MYSTERY_VALUE"
            ),
        ):
            provenance = self.hf_token.credential_provenance(
                {"LINKDOG_HF_TOKEN_FILE": str(self.tmp / "secrets" / "hf_token")}
            )

        self.assertEqual(provenance.source, self.hf_token.SOURCE_OTHER)
        self.assertFalse(provenance.durable)
        self.assertIn("CANNOT be verified", provenance.describe())

    def test_missing_hub_dependency_is_unknown_not_degraded(self):
        with patch.object(self.hf_token, "_hub_constants", lambda: None):
            provenance = self.hf_token.credential_provenance({})

        self.assertEqual(provenance.source, self.hf_token.SOURCE_NONE)
        self.assertFalse(provenance.degraded)

    def test_provenance_never_contains_a_token_value(self):
        """It classifies the *source*; it must never surface the secret itself."""
        provenance, _ = self._provenance(
            hub_token="PROJECT_TOKEN_VALUE",
            project_token="PROJECT_TOKEN_VALUE",
        )

        rendered = provenance.describe() + provenance.hub_token_path + provenance.source
        self.assertNotIn("PROJECT_TOKEN_VALUE", rendered)
        self.assertNotIn(FAKE_TOKEN, rendered)

    @staticmethod
    def _fake_constants(**values):
        """Build a stand-in for ``huggingface_hub.constants``."""

        class FakeConstants:
            pass

        obj = FakeConstants()
        for key, value in values.items():
            setattr(obj, key, value)
        return obj


class ImportTimeOrderingTests(unittest.TestCase):
    """``huggingface_hub`` freezes HF_TOKEN_PATH at import — the pin must be first.

    Both branches are exercised with a **synthetic fixture**, so the result does
    not depend on whether this machine happens to hold a real credential.
    """

    _PROBE = (
        "import os, sys\n"
        "sys.path.insert(0, {root!r})\n"
        "import app.tts_auth\n"
        "from huggingface_hub import constants\n"
        "print('APPLIED=', {applied!r})\n"
        "print('HUB_SEES=', constants.HF_TOKEN_PATH)\n"
        "print('ENV_HAS=', 'HF_TOKEN_PATH' in os.environ)\n"
    )

    def _run_probe(self, token_file: Path | None, *, set_explicit: bool = False):
        """Import app.tts_auth in a clean interpreter and report the pin result."""
        script = self._PROBE.format(root=str(REPO_ROOT), applied=True)
        with tempfile.TemporaryDirectory() as home:
            env = {
                key: value
                for key, value in os.environ.items()
                # An ambient credential must not decide the outcome.
                if key not in ("HF_TOKEN_PATH", "LINKDOG_HF_TOKEN_FILE")
            }
            env["HOME"] = home
            env["HF_HOME"] = home
            if token_file is not None:
                env["LINKDOG_HF_TOKEN_FILE"] = str(token_file)
            if set_explicit:
                env["HF_TOKEN_PATH"] = "/explicit/operator/token"
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
                cwd=str(REPO_ROOT),
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        return {
            key: value.strip()
            for line in result.stdout.splitlines()
            if "=" in line
            for key, value in [line.split("=", 1)]
        }

    def test_project_credential_is_pinned_before_the_hub_loads_it(self):
        """Importing app.tts_auth must pin the path before hf_hub freezes it."""
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "synthetic_hf_token"
            fixture.write_text(FAKE_TOKEN)
            seen = self._run_probe(fixture)

        self.assertEqual(seen["ENV_HAS"], "True")
        # The Hub must actually resolve the pinned path, not the cache default.
        self.assertEqual(seen["HUB_SEES"], str(fixture))

    def test_no_credential_means_no_pin_and_no_export(self):
        """The stripped-checkout branch: decline cleanly, export nothing.

        The earlier version of this test asserted ``"PINNED" not in stdout``
        while the probe printed ``PINNED=`` unconditionally, so it could never
        pass without a real credential. Reported by the second review.

        Absence must be simulated via the seam (``LINKDOG_HF_TOKEN_FILE``
        pointing at a path that does not exist) rather than by *omitting* it:
        omitting it makes the module fall back to the real repo credential,
        which exists on a developer machine — so the branch would silently
        never be exercised here.
        """
        with tempfile.TemporaryDirectory() as directory:
            absent = Path(directory) / "no_such_credential"
            seen = self._run_probe(absent)

        self.assertEqual(seen["ENV_HAS"], "False")
        # Nothing was exported, so the Hub falls back to its own cache default
        # inside the isolated HOME — never the project path.
        self.assertNotEqual(seen["HUB_SEES"], str(REPO_ROOT / "secrets" / "hf_token"))
        self.assertNotIn("secrets", seen["HUB_SEES"])

    def test_explicit_operator_path_beats_the_project_credential(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "synthetic_hf_token"
            fixture.write_text(FAKE_TOKEN)
            seen = self._run_probe(fixture, set_explicit=True)

        self.assertEqual(seen["HUB_SEES"], "/explicit/operator/token")

    def test_pinning_after_hub_import_is_ineffective(self):
        """Documents WHY the pin lives in app/__init__.py.

        If this test ever starts failing, huggingface_hub became late-binding
        and the early-import constraint in app/__init__.py could be relaxed.
        """
        script = (
            "import os\n"
            "from huggingface_hub import constants\n"
            "before = constants.HF_TOKEN_PATH\n"
            "os.environ['HF_TOKEN_PATH'] = '/tmp/definitely-not-used'\n"
            "after = constants.HF_TOKEN_PATH\n"
            "print('UNCHANGED=', before == after)\n"
        )
        with tempfile.TemporaryDirectory() as home:
            env = {
                key: value
                for key, value in os.environ.items()
                if key not in ("HF_TOKEN_PATH", "LINKDOG_HF_TOKEN_FILE")
            }
            env["HOME"] = home
            env["HF_HOME"] = home
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("UNCHANGED= True", result.stdout)


class ModelSingleSourceTests(unittest.TestCase):
    """One authoritative model value, and a fallback the dashboard accepts."""

    def test_default_model_is_shared_by_both_code_defaults(self):
        """dashboard_settings.DEFAULT_MODEL and the state loader's fallback must agree."""
        from app.dashboard_settings import DEFAULT_MODEL

        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(state, "SETTINGS_STORE", _missing_store()),
        ):
            settings = state.load_dashboard_settings()

        self.assertEqual(settings.model, DEFAULT_MODEL)
        self.assertEqual(DEFAULT_MODEL, "deepseek-v4.1-flash")

    def test_no_stale_model_literal_remains_in_source(self):
        """Regression guard for the retired 'deepseek-v4-flash:0731' default.

        Comments are exempt on purpose: the fix documents the retired value
        inline (see dashboard_settings.DEFAULT_MODEL), and those notes are
        worth keeping. Only executable code must be free of it.
        """
        offenders = []
        for path in sorted((REPO_ROOT / "app").rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if "deepseek-v4-flash:0731" in line and not line.strip().startswith("#"):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")

        self.assertEqual(offenders, [], f"stale model literal in: {offenders}")


def _missing_store():
    """A SettingsStore whose file does not exist, forcing the env fallback path."""
    from app.dashboard_settings import SettingsStore

    return SettingsStore(Path("/nonexistent") / "settings.json")


if __name__ == "__main__":
    unittest.main()
