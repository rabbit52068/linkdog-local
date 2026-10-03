#!/bin/bash
# Start hermes-linkdog adapter (production, voice disabled by default)
set -a
source "$(dirname "$0")/.env" || { echo "[run_adapter] failed to source .env" >&2; exit 1; }
set +a
cd "$(dirname "$0")" || { echo "[run_adapter] failed to cd to script dir" >&2; exit 1; }

# Fail-closed TTS preflight: abort before uvicorn starts if the configured
# Pocket TTS voice is a local file that needs voice cloning but cloning is
# unavailable, or a .safetensors state that cannot be imported. The backend
# skip decision lives entirely in Python (scripts/tts_preflight.py) so the
# shell never parses LINKDOG_TTS_BACKEND and cannot be bypassed via
# whitespace/case tricks (Astra blocking #1/#4).
if ! .venv-dev/bin/python scripts/tts_preflight.py; then
    echo "[run_adapter] TTS preflight failed; refusing to start." >&2
    exit 1
fi

exec .venv-dev/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8003
