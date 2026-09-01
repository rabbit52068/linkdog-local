# LinkDog Local — Project Plan

> Self-hosted adapter for the LinkDog robot dog: replace the official cloud with your own LLM, keep official WeChat mini-program control.

## Status

Planning complete. Repo scaffolded with `PLAN.md` + `README.md` only; source code is not yet committed.

## Goal

Let other LinkDog owners self-host the adapter, plug in any OpenAI-compatible / Ollama LLM, and keep official provisioning + WeChat mini-program control.

## Architecture

```
LinkDog (ESP32-S3 + C3)
   │  Provisioning: official WeChat mini-program (Bluetooth, writes wifi creds)
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

## Verified Facts

- **mDNS is feasible** (spike done): `CONFIG_LWIP_DNS_SUPPORT_MDNS_QUERIES=y` alone lets ESP32 resolve `.local` — no esp_mdns component, no custom query. lwIP sends the query to multicast `224.0.0.251:5353`; the responder replies via unicast (RFC 6762). Prerequisite: adapter host runs a standard mDNS responder (avahi/Bonjour/esp_mdns). Change is a pure string replace (hard-coded IP → `linkdog.local`).
- **WeChat mini-program controls via Bluetooth direct** (verified by owner) — no wifi, no adapter.
- **Multi-dog**: adapter already supports it (`ACTIVE_SESSIONS` dict keyed by MAC). mDNS is for portability, not multi-dog disambiguation.
- **License**: firmware source is MIT (LinkDog vendor). esp-sr `.a` libs are ESPRESSIF MIT — redistributable, but prefer `idf_component.yml` dependency over committing binaries.

## Decisions

- Single repo: adapter + firmware source + pre-built firmware bin (bin via GitHub Release, not git history).
- mDNS hostname: single `linkdog.local`.
- License: MIT.
- LLM config in dashboard (approach C): non-secret fields (model/provider/api_url) → `data/settings.json`; api_key → `.env`.
- Restart stays agent-managed (service/docker residency undecided).
- Source code: English-only comments.

## Roadmap

- **Phase 0 — Verification**: DONE (mDNS feasible, WeChat Bluetooth confirmed).
- **Phase 1 — Privacy cleanup**: DONE (`.gitignore`, `settings.example.json`, hard-coded values replaced with auto-detect/empty).
- **Phase 2 — Config externalization**: HOST auto-detect, DEVICE_ID empty, dashboard LLM config.
- **Phase 3 — Documentation**: README, new-dog setup flow.
- **Phase 4 — Portability**: complete requirements, Docker/setup.sh, multi-device.
- **Phase 5 — Firmware changes** (irreversible): mDNS (3 hard-coded IPs) + remove `liblinkdog_mqtt.a` closed-source dep.
- **Phase 6 — Chinese comment cleanup**: 99 lines across 6 files.
