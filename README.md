# LinkDog Local

Self-hosted adapter for the [LinkDog](https://gitee.com/jeremywang0102/linkdog) robot dog. Replace the official cloud with your own LLM while keeping official WeChat mini-program control.

> **Status: early development.** The adapter works end-to-end, but setup still requires some manual steps (see [New dog setup](#new-dog-setup)). Contributions and feedback welcome.

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
                                 ├─ ASR (faster-whisper)
                                 ├─ LLM (Ollama Cloud / any OpenAI-compatible)
                                 ├─ TTS (Pocket TTS / edge-tts)
                                 └─ MCP bridge → motion control
```

Three paths are fully decoupled: swapping the adapter only affects voice/LLM/OTA; provisioning and control are untouched.

## Requirements

- A LinkDog robot dog (ESP32-S3 main + C3 co-processor).
- A computer on the same LAN to run the adapter (macOS, Linux, or Windows).
- Python 3.11+.

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

## New dog setup

1. **Provision** — use the official WeChat mini-program to connect the dog to your Wi-Fi (Bluetooth, writes credentials).
2. **Get the MAC** — the dog's MAC address identifies it to the adapter (shown in the dashboard once connected).
3. **Point to the adapter** — the dog's OTA/WebSocket URLs must resolve to the adapter host. The stock firmware points at a hard-coded IP; see [Firmware](#firmware) for how to point it at your adapter.
4. **Verify** — the dog connects over WebSocket; check the dashboard shows it as connected and test a voice command.

## Firmware

The stock firmware hard-codes the adapter host as a LAN IP. To use this adapter you need to rebuild the firmware with your host address (or a `.local` mDNS name). The firmware source is MIT and available from the [upstream LinkDog repo](https://gitee.com/jeremywang0102/linkdog).

- **mDNS** *(recommended)*: enable `CONFIG_LWIP_DNS_SUPPORT_MDNS_QUERIES=y` and replace the hard-coded IP with `linkdog.local`. Your adapter host must run a standard mDNS responder (avahi/Bonjour/esp_mdns). This keeps the firmware portable across networks.
- **IP direct**: replace the hard-coded IP with your adapter host's LAN IP. Simpler, but breaks if your IP changes.

## Configuration

- **LLM**: non-secret fields (model, provider, API URL) live in `data/settings.json`; the API key stays in `.env`.
- **Multiple dogs**: the adapter supports several dogs, keyed by MAC address.

## License

MIT. The upstream LinkDog firmware source is also MIT.
