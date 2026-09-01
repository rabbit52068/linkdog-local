#!/bin/bash
# Start hermes-linkdog adapter (production, voice disabled by default)
set -a
source "$(dirname "$0")/.env"
set +a
cd "$(dirname "$0")"
exec .venv-dev/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8003
