"""Mask secrets before exception text reaches logs or ``/health``.

A TTS load failure surfaces the underlying exception message, and HF/Hub
errors routinely embed the *request URL* — which can carry a ``?token=...``
query parameter. That text is published by ``/api/health`` and shown on the
dashboard, so a credential could end up in a browser, a log
aggregator, or a screenshot.

The rule applied here is deliberately blunt: it is better to over-mask a
diagnostic string than to leak a credential. Only the secret *values* are
replaced; the surrounding message stays readable so operators can still tell
what failed.

Architecture
------------
An earlier per-scheme patch (adding ``bearer|token|basic`` to the keyed rule)
still leaked six of eight credential shapes. The failures shared two
structural causes:

1. **The key rule assumed bare ``key=value``.** A JSON-quoted key
   (``{"Authorization": "Basic ..."}``) has a quote between the key and the
   separator, so the key rule never matched and the value survived untouched.
2. **The value class was too narrow, and schemes were enumerated.** Values
   containing ``=`` or ``/`` were masked only up to that character, leaving the
   secret's tail behind. Enumerating schemes meant every unlisted one
   (``Negotiate``, ``Digest``, AWS's ``AWS4-HMAC-SHA256``) fell through to
   whatever rule happened to be next.

Round-3 review found five further defects, all reproduced before fixing:

1. **The scheme class was narrower than RFC 7235.** A scheme is a ``token``,
   which permits ``!#$%&'*+-.^_`|~``; the old class allowed only
   ``[A-Za-z0-9._-]``, so ``Authorization: Custom+Auth <secret>`` leaked.
   The class is now the RFC token set (minus quote characters, which belong to
   the value/quoting logic).
2. **Only the first parameter of an unknown scheme was masked.** A header like
   ``Authorization: Custom "<s>", second="<s>"`` lost only its first value,
   because parameter-list handling was hard-wired to the Digest/AWS4/OAuth
   enumeration. An auth header now owns the **rest of its line**, and every
   ``name=value`` in that tail is masked — so unknown schemes are covered by
   construction rather than by being listed.
3. **An escaped quote truncated a JSON string value.** The quoted alternative
   was ``"[^"]*"``, so ``{"password": "prefix\\"<secret>"}`` matched only up to
   the escape and left the secret's tail in place. Quoted values are now
   escape-aware.
4. **Masking could invalidate JSON.** Both key rules wrote the optional
   post-key quote with ``["']?`` inside the *separator* group and then re-emitted
   only ``key`` + ``separator`` — silently dropping the key's closing quote
   (``{"Authorization: "[REDACTED]"}``, which ``json.loads`` rejects). The quote
   is now its own capture group and is re-emitted, so a masked JSON document
   stays parseable.
5. **URL userinfo credentials were not handled at all.** ``HF`` errors print
   URLs, and ``https://user:<secret>@host/path`` carries a credential in the
   authority section, which no keyed rule covers. Those passwords are masked.

This version classifies by *header shape* instead:

* ``_URL_USERINFO_RE`` — ``scheme://user:password@host`` credentials.
* ``_AUTH_HEADER_RE`` — any RFC 7235 auth-scheme word before the credential
  (so ``Negotiate``, ``NTLM``, ``Custom+Auth`` and future schemes are covered by
  construction, not by enumeration); it also owns its line's parameter tail.
* ``_MULTIPARAM_SCHEME_RE`` — Digest / AWS4 / OAuth carry several
  ``name=value`` parameters, all of which are credential material; the whole
  parameter list is masked.
* ``_GENERIC_SECRET_RE`` — ordinary ``key=value``/``key: value`` secrets.

Idempotency is structural: ``[REDACTED]`` can never re-match any value
alternative (bracket characters are excluded from bare values, and a quoted
``"[REDACTED]"`` re-masks to itself), so a second pass is always a no-op.
"""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"

# Auth-header names. These are matched by `_AUTH_HEADER_RE`, which tolerates an
# arbitrary scheme word, rather than by the generic rule.
_AUTH_HEADER_KEYS = r"authorization|proxy[_-]?authorization|x[_-]?auth|bearer|auth"

# Ordinary secret keys: `api_key=...`, `password: ...`, `?token=...`.
_GENERIC_SECRET_KEYS = (
    r"access[_-]?token|refresh[_-]?token|id[_-]?token|csrf[_-]?token|"
    r"x[_-]?api[_-]?key|api[_-]?key|apikey|client[_-]?secret|"
    r"secret|password|passwd|pwd|sig|signature|token"
)

# Schemes that carry a parameter LIST rather than one opaque value. Owned by
# `_MULTIPARAM_SCHEME_RE`; explicitly excluded from `_AUTH_HEADER_RE` so the
# two rules cannot fight over the same span.
_MULTIPARAM_SCHEMES = r"digest|aws4-hmac-sha256|oauth"

# An auth-scheme is an RFC 7235 `token`. Quote characters are deliberately NOT
# included: they are structural here (they delimit the value), and allowing them
# inside the scheme word would let the scheme swallow its own quoting.
_SCHEME_CHARS = r"[A-Za-z][A-Za-z0-9!#$%&*+\-.^_`|~]*"

# A quoted string, escape-aware. `\\.` consumes an escaped quote (or any other
# escaped character) so a value such as `"prefix\"secret"` is matched whole
# instead of being cut off at the first inner quote.
_QUOTED = r'"(?:[^"\\]|\\.)*"' + r"|'(?:[^'\\]|\\.)*'"

# Value alternation. Order matters:
#   1. any quoted string (keeps a JSON/dict rendering valid, and consumes
#      escaped quotes correctly),
#   2. a bare literal ``[REDACTED]`` — REQUIRED for idempotency. The bare-value
#      class below excludes brackets, so without this alternative an
#      already-masked line could not be consumed as a value at all; the key rule
#      then backtracked, dropped its optional scheme group, and masked the
#      *scheme word* instead (masking the same header twice with a different
#      result each pass).
#   3. a bare token, allowing ``=``, ``/`` and ``+`` so `token=abcd=<secret>`
#      and `Bearer xy/<secret>` are consumed whole, while bracket characters and
#      quote/separator characters stay excluded.
# A quoted ``"[REDACTED]"`` is matched by alternative 1 and re-emitted unchanged
# by `_mask_value`, which is what keeps the whole function idempotent.
_VALUE = (
    _QUOTED
    + "|"
    + re.escape(REDACTED)
    + r"|[^\s&,;'\")\[\]{}]+"
)

# An auth header: optional quote after the key (JSON), any scheme word, value,
# then the REST OF THE LINE as `tail`. The tail is what makes unknown schemes
# with parameter lists safe: `_mask_auth_header` masks every `name=value` in it,
# so `Authorization: Custom "<s>", second="<s>"` cannot lose the second value.
_AUTH_HEADER_RE = re.compile(
    r"(?i)\b("
    + _AUTH_HEADER_KEYS
    + r")\b([\"']?)(\s*[:=]\s*)"
    # Reject multi-parameter schemes outright; the next rule owns them.
    + r"(?!(?:"
    + _MULTIPARAM_SCHEMES
    + r")\b)"
    + r"(?:(?P<scheme>"
    + _SCHEME_CHARS
    + r")\s+)?"
    + r"(?P<value>"
    + _VALUE
    + r")"
    + r"(?P<tail>[^\n]*)"
)

# Digest / AWS4 / OAuth: every `name=value` in the parameter list is credential
# material (nonce, response, cnonce, signature, ...). Only the *values* go.
_PARAM_RE = re.compile(
    r"(?i)\b([A-Za-z][A-Za-z0-9_\-]*)(\s*=\s*)(" + _QUOTED + r"|[^\s,;]+)"
)

_MULTIPARAM_SCHEME_RE = re.compile(
    r"(?i)\b(?:" + _MULTIPARAM_SCHEMES + r")\b([^\n]*)"
)

# Ordinary keyed secrets, optionally carrying a single-value scheme. The
# post-key quote is captured separately so it can be re-emitted (see defect 4).
_GENERIC_SECRET_RE = re.compile(
    r"(?i)\b("
    + _GENERIC_SECRET_KEYS
    + r")\b([\"']?)(\s*[:=]\s*)"
    r"(?:(bearer|basic|token)\s+)?"
    r"("
    + _VALUE
    + r")"
)

# Credentials embedded in a URL authority: `scheme://user:password@host`.
# HF/Hub error messages routinely print the request URL, and the token can sit
# in the userinfo section where no keyed rule would find it. The user name is
# kept (it is not the secret and it aids debugging); the password is masked.
_URL_USERINFO_RE = re.compile(
    r"(?i)\b([a-z][a-z0-9+.\-]*://)([^/\s:@]+):([^/\s@]+)@"
)

# A bare ``Bearer <token>`` header value with no ``=``. Covers the form that has
# no key at all, and is the last-resort net for token shapes containing ``/``,
# ``+`` or ``=`` (base64 padding).
_BEARER_RE = re.compile(
    r"(?i)\bbearer\s+(?!\[REDACTED\])[A-Za-z0-9._\-/+=]{8,}"
)

# Known credential *shapes*, which are masked even when they appear without a
# key (e.g. inside a URL path or an interpolated message).
_TOKEN_SHAPE_RE = re.compile(
    r"\b(?:hf_|sk-|sk_|ghp_|gho_|ghu_|ghs_|github_pat_|xox[baprs]-)"
    r"[A-Za-z0-9_\-]{8,}"
)

# JWT-ish triples (three base64url segments) which carry no distinguishing
# prefix and are otherwise indistinguishable from ordinary dotted text.
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{6,}")


def _mask_value(value: str) -> str:
    """Return ``[REDACTED]``, re-using the original quoting if any.

    Preserving the quote characters keeps a masked line parseable (a JSON or
    Python dict rendering stays valid) and keeps the redaction idempotent: a
    quoted ``"[REDACTED]"`` masks to itself rather than accumulating quotes.
    """
    if value[:1] in ('"', "'"):
        return f"{value[0]}{REDACTED}{value[0]}"
    return REDACTED


def _mask_params(text: str) -> str:
    """Mask the value of every ``name=value`` pair in ``text``."""
    return _PARAM_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{_mask_value(m.group(3))}", text
    )


def _mask_userinfo(match: re.Match) -> str:
    """Keep scheme and user, mask the password in a URL authority."""
    return f"{match.group(1)}{match.group(2)}:{REDACTED}@"


def _mask_auth_header(match: re.Match) -> str:
    """Re-emit key, its quote, separator and scheme; never the value.

    The line tail is masked too: an auth header owns its parameter list, which
    is how schemes we never enumerated stay covered.
    """
    scheme = match.group("scheme")
    return (
        f"{match.group(1)}{match.group(2)}{match.group(3)}"
        f"{scheme + ' ' if scheme else ''}"
        f"{_mask_value(match.group('value'))}"
        f"{_mask_params(match.group('tail') or '')}"
    )


def _mask_generic_secret(match: re.Match) -> str:
    """Re-emit key, its quote, separator and optional scheme; never the value."""
    scheme = match.group(4)
    return (
        f"{match.group(1)}{match.group(2)}{match.group(3)}"
        f"{scheme + ' ' if scheme else ''}"
        f"{_mask_value(match.group(5))}"
    )


def _mask_multiparam(match: re.Match) -> str:
    """Mask every ``name=value`` parameter after a multi-parameter scheme.

    A Digest/Negotiate/AWS4 tail is credential material end to end, so all
    values are masked. When the tail carries no parameter list at all the match
    is returned unchanged — that keeps this rule harmless on prose that merely
    contains the word "digest". (Computing the head as ``[:-len(tail)]`` broke
    exactly that case: an empty tail made the slice ``[:-0]``, i.e. empty, so
    the whole match was replaced by ``""``.)
    """
    whole = match.group(0)
    tail = match.group(1)
    head = whole[: len(whole) - len(tail)]
    return head + _mask_params(tail)


def redact_secrets(text: str) -> str:
    """Return ``text`` with credential-looking values replaced by ``[REDACTED]``.

    Idempotent (re-running over already-redacted text is a no-op) and safe on
    empty/None input (``None`` -> ``""``).
    """
    if not text:
        return ""
    result = str(text)
    result = _JWT_RE.sub(REDACTED, result)
    # URL userinfo first: the credential sits in the authority section, where
    # the keyed rules have no key to anchor on.
    result = _URL_USERINFO_RE.sub(_mask_userinfo, result)
    # Multi-parameter auth schemes next: they own their whole parameter list,
    # and `_AUTH_HEADER_RE` deliberately refuses to match them.
    result = _MULTIPARAM_SCHEME_RE.sub(_mask_multiparam, result)
    result = _AUTH_HEADER_RE.sub(_mask_auth_header, result)
    result = _GENERIC_SECRET_RE.sub(_mask_generic_secret, result)
    result = _BEARER_RE.sub(f"Bearer {REDACTED}", result)
    result = _TOKEN_SHAPE_RE.sub(REDACTED, result)
    return result
