"""hermes-linkdog adapter package.

Import-time side effect: pin ``HF_TOKEN_PATH`` to the project-local credential
file *before* anything can import ``huggingface_hub``.

This has to happen here rather than in ``app/main.py`` or ``app/tts_auth.py``
because ``huggingface_hub`` freezes its token path into
``constants.HF_TOKEN_PATH`` at import time — setting the environment variable
afterwards is silently ignored. Importing any ``app.*`` submodule executes this
package init first, which makes it the only choke point that is guaranteed to
run before the Hub is loaded.

The call is guarded and non-fatal: it defers to an explicit ``HF_TOKEN_PATH``,
does nothing when the project file is absent or blank, and never raises. See
:mod:`app.hf_token` for the full rationale and the 2026-09-13 failure it
prevents. The token value is never read or logged here — only the path moves.
"""

from app.hf_token import pin_hf_token_path as _pin_hf_token_path

_pin_hf_token_path()
