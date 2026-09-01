# LinkDog Local

Self-hosted adapter for the [LinkDog](https://www.linkdog.com) robot dog. Replace the official cloud with your own LLM while keeping official WeChat mini-program control.

> **Status: early development.** Source code is being prepared for release and is not yet published in this repository. This README documents the target architecture and setup flow.

## What it does

- Runs a local FastAPI adapter that the LinkDog connects to over WebSocket.
- Speech pipeline: faster-whisper (ASR) → your LLM (any OpenAI-compatible / Ollama) → Pocket TTS / edge-tts.
- Motion control via the official firmware MCP bridge.
- Web dashboard to configure role, model, memory, and volume.
- Keeps official provisioning (WeChat mini-program over Bluetooth) and Bluetooth control untouched.

## Architecture

```
LinkDog (ESP32-S3 + C3)
   │  Provisioning: WeChat mini-program (Bluetooth, writes wifi creds)
   │  Control: WeChat mini-program (Bluetooth direct — no wifi, no adapter)
   │
   └── Voice / LLM / OTA ──►  adapter (FastAPI :8003)
                                 │  mDNS: linkdog.local
                                 ├─ ASR (faster-whisper)
                                 ├─ LLM (Ollama Cloud / any OpenAI-compatible)
                                 ├─ TTS (Pocket TTS / edge-tts)
                                 └─ MCP bridge → motion control
```

Three paths are fully decoupled: swapping the adapter only affects voice/LLM/OTA; provisioning and control are untouched.

## Hardware requirements

- A LinkDog robot dog (ESP32-S3 main + C3 co-processor).
- A computer on the same LAN to run the adapter (macOS, Linux, or Windows).
- Python 3.11+.

## Dependencies

Core (see `requirements.txt`):

- `fastapi` + `uvicorn[standard]` — HTTP/WebSocket server.
- `faster-whisper` — speech-to-text (ASR).
- `webrtcvad-wheels` — voice activity detection.
- `edge-tts` — cloud TTS fallback.
- `numpy` — audio processing (ASR path).

MCP bridge (see `requirements-mcp.txt`):

- `fastmcp` + `httpx` — motion-control MCP server.

Optional:

- Pocket TTS (local neural TTS) — `pocket-tts`, `torch`, `scipy`, `soundfile`. See `requirements-tts.txt` and `docs/pocket-tts.md`.

## Setup

1. Clone the repository.
2. Create a virtual environment and install dependencies:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and fill in your values (API key, model, TTS backend, etc.).
4. Start the adapter:

   ```bash
   ./run_adapter.sh
   ```

   The adapter listens on port `8003` and serves the dashboard at `http://<host>:8003/dashboard`.

## New dog setup

1. **Provision** — use the official WeChat mini-program to connect the dog to your Wi-Fi (Bluetooth, writes credentials).
2. **Get the MAC** — the dog's MAC address identifies it to the adapter (shown in the dashboard once connected).
3. **Point to the adapter** — the dog's OTA/WebSocket URLs resolve to the adapter host (see Design Notes on mDNS).
4. **Verify** — the dog connects over WebSocket; check the dashboard shows it as connected and test a voice command.

## Design Notes

- **mDNS** *(planned — see Roadmap)*: `CONFIG_LWIP_DNS_SUPPORT_MDNS_QUERIES=y` alone lets the ESP32 resolve `.local` — no esp_mdns component, no custom query. lwIP sends the query to multicast `224.0.0.251:5353`; the responder replies via unicast (RFC 6762). The adapter host must run a standard mDNS responder (avahi/Bonjour/esp_mdns). Firmware change is a pure string replace (hard-coded IP → `linkdog.local`).
- **Multi-dog**: the adapter supports multiple dogs (`ACTIVE_SESSIONS` keyed by MAC). mDNS is for portability, not multi-dog disambiguation.
- **LLM config**: non-secret fields (model/api_url) live in `data/settings.json`; the API key stays in `.env`.

## Roadmap

- **mDNS firmware change** — replace hard-coded IPs with `linkdog.local` (requires firmware rebuild + flash).
- **Remove closed-source MQTT dependency** — drop `liblinkdog_mqtt.a` (zero functional loss).
- **Portability** — complete `requirements.txt`, consider Docker.

## License

MIT. Firmware source is MIT (LinkDog vendor); esp-sr `.a` libs are ESPRESSIF MIT (prefer `idf_component.yml` dependency over committing binaries).
