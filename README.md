# LinkDog Local

Self-hosted adapter for the [LinkDog](https://www.linkdog.com) robot dog. Replace the official cloud with your own LLM while keeping official WeChat mini-program control.

> **Status: early development.** This repository currently contains only the project plan and this README. Source code will be added as it is prepared for release.

## What it does

- Runs a local FastAPI adapter that the LinkDog connects to over WebSocket.
- Speech pipeline: faster-whisper (ASR) → your LLM (any OpenAI-compatible / Ollama) → Pocket TTS / edge-tts.
- Motion control via the official firmware MCP bridge.
- Web dashboard to configure role, model, memory, and volume.
- Keeps official provisioning (WeChat mini-program over Bluetooth) and Bluetooth control untouched.

## Architecture

```
LinkDog (ESP32-S3 + C3)
   │  Provisioning: WeChat mini-program (Bluetooth)
   │  Control: WeChat mini-program (Bluetooth direct)
   │
   └── Voice / LLM / OTA ──►  adapter (FastAPI :8003)
                                 │  mDNS: linkdog.local
                                 ├─ ASR (faster-whisper)
                                 ├─ LLM (Ollama Cloud / OpenAI-compatible)
                                 ├─ TTS (Pocket TTS / edge-tts)
                                 └─ MCP bridge → motion control
```

## License

MIT. See the project plan (`PLAN.md`) for firmware and third-party component licensing notes.

## Status

See `PLAN.md` for the full roadmap and current progress.
