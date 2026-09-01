# LinkDog Local

Self-hosted adapter for the [LinkDog](https://www.linkdog.com) robot dog. Replace the official cloud with your own LLM while keeping official WeChat mini-program control.

> **Status: early development.** This repository currently contains only this README. Source code will be added as it is prepared for release.

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

## Design Notes

- **mDNS**: `CONFIG_LWIP_DNS_SUPPORT_MDNS_QUERIES=y` alone lets the ESP32 resolve `.local` — no esp_mdns component, no custom query. lwIP sends the query to multicast `224.0.0.251:5353`; the responder replies via unicast (RFC 6762). The adapter host must run a standard mDNS responder (avahi/Bonjour/esp_mdns). Firmware change is a pure string replace (hard-coded IP → `linkdog.local`).
- **Multi-dog**: the adapter supports multiple dogs (`ACTIVE_SESSIONS` keyed by MAC). mDNS is for portability, not multi-dog disambiguation.
- **LLM config**: non-secret fields (model/provider/api_url) live in `data/settings.json`; the API key stays in `.env`.

## License

MIT. Firmware source is MIT (LinkDog vendor); esp-sr `.a` libs are ESPRESSIF MIT (prefer `idf_component.yml` dependency over committing binaries).
