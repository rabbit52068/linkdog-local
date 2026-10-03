#!/bin/bash
# Start hermes-linkdog adapter (production, voice disabled by default)
set -a
source "$(dirname "$0")/.env" || { echo "[run_adapter] failed to source .env" >&2; exit 1; }
set +a
cd "$(dirname "$0")" || { echo "[run_adapter] failed to cd to script dir" >&2; exit 1; }

# When LINKDOG_LOG_FILE is set (e.g. by the LaunchAgent), append all output to
# it, rotating once it passes 10 MB so the log cannot grow without bound.
if [ -n "${LINKDOG_LOG_FILE:-}" ]; then
    mkdir -p "$(dirname "$LINKDOG_LOG_FILE")"
    if [ -f "$LINKDOG_LOG_FILE" ] && [ "$(stat -f%z "$LINKDOG_LOG_FILE")" -gt 10485760 ]; then
        mv -f "$LINKDOG_LOG_FILE" "$LINKDOG_LOG_FILE.1"
    fi
    exec >>"$LINKDOG_LOG_FILE" 2>&1
    echo "[run_adapter] starting at $(date '+%Y-%m-%d %H:%M:%S')"
fi
export PYTHONUNBUFFERED=1

# Fail-closed TTS preflight: abort before uvicorn starts if the configured
# Pocket TTS voice is a local file that needs voice cloning but cloning is
# unavailable, or a .safetensors state that cannot be imported. The backend
# skip decision lives entirely in Python (scripts/tts_preflight.py) so the
# shell never parses LINKDOG_TTS_BACKEND and cannot be bypassed via
# whitespace/case tricks.
if ! .venv-dev/bin/python scripts/tts_preflight.py; then
    echo "[run_adapter] TTS preflight failed; refusing to start." >&2
    exit 1
fi

exec .venv-dev/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8003
