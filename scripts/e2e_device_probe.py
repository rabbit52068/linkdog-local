"""LinkDog simulated-device end-to-end probe (adapter side, no real dog needed).

Simulates an ESP32-S3 device against the local adapter:
  WS hello -> listen:start -> upload OPCUS frames -> receive tts:start / llm emotion /
  sentence_start / downlink audio -> decode Opus back to PCM -> ASR feedback -> tts:stop.

This exercises the whole adapter voice chain (VAD endpoint -> ASR -> LLM -> TTS ->
Opus downlink) without needing the physical dog, which only opens its WebSocket after
a wake word. It also verifies the "Chinese in -> English out" contract end to end.

Usage:
  .venv-dev/bin/python scripts/e2e_device_probe.py --wav /tmp/e2e_utt_padded.wav
  .venv-dev/bin/python scripts/e2e_device_probe.py --say "小斌，今天天氣怎麼樣？"

  # --say generates a padded 16k mono wav with `say` + ffmpeg (see _build_say_wav).
  # Padded wavs already on disk: /tmp/e2e_utt_padded.wav (EN), /tmp/e2e_zh_padded.wav (ZH).

Read-only with respect to production state: it only connects to the local adapter and
writes decoded audio under --out-dir. It never touches data/settings.json or the repo.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import soundfile as sf
import websockets

from app.audio_codec import OpusCodec

# Fake MAC: never reuse the real device id, or this probe would hijack the live session.
DEVICE_ID = "02:00:00:00:00:e2"
SAMPLE_RATE = 16000
LEAD_SILENCE_S = 1.0   # clear the 800ms wake-word discard window (13 frames @ 60ms)
TAIL_SILENCE_S = 1.6   # must exceed the 440ms VAD silence threshold or no endpoint fires


def _build_say_wav(text: str, out_path: Path, voice: str | None = None) -> Path:
    """Render `text` with macOS say and pad it with silence on both ends.

    Padding is not cosmetic: without leading silence the pipeline's wake-word discard
    window eats the first ~800ms; without trailing silence the VAD never reaches its
    silence threshold, so no utterance is ever endpointed and ASR is never called.
    """
    aiff = out_path.with_suffix(".aiff")
    say = ["say"]
    if voice:
        say += ["-v", voice]
    say += [text, "-o", str(aiff)]
    subprocess.run(say, check=True)

    raw_wav = out_path.with_name(out_path.stem + "_raw.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff),
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(raw_wav)],
        check=True,
    )
    audio, rate = sf.read(str(raw_wav), dtype="int16", always_2d=False)
    assert rate == SAMPLE_RATE, f"unexpected sample rate {rate}"
    pad = lambda s: np.zeros(int(s * SAMPLE_RATE), dtype="int16")  # noqa: E731
    padded = np.concatenate([pad(LEAD_SILENCE_S), audio, pad(TAIL_SILENCE_S)])
    sf.write(str(out_path), padded, SAMPLE_RATE)
    raw_wav.unlink(missing_ok=True)
    aiff.unlink(missing_ok=True)
    rms = float(np.sqrt(np.mean(audio.astype("float64") ** 2))) if audio.size else 0.0
    print(f"[probe] say wav: {out_path} ({len(padded)/SAMPLE_RATE:.2f}s, "
          f"speech {len(audio)/SAMPLE_RATE:.2f}s, RMS={rms:.0f})")
    return out_path


def load_pcm_16k_mono(path: str) -> bytes:
    audio, rate = sf.read(path, dtype="int16", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype("int16")
    if rate != SAMPLE_RATE:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(rate, SAMPLE_RATE)
        audio = resample_poly(
            audio.astype("float32"), SAMPLE_RATE // g, rate // g
        ).astype("int16")
    return audio.tobytes()


async def run(url: str, pcm: bytes, text_note: str, idle_wait: float, out_dir: Path) -> int:
    codec = OpusCodec(sample_rate=SAMPLE_RATE, channels=1, frame_duration_ms=60)
    frame_bytes = codec.pcm_bytes_per_frame
    total_frames = len(pcm) // frame_bytes
    print(f"[probe] PCM {len(pcm)} bytes = {total_frames} frames of 60ms "
          f"({len(pcm)/2/SAMPLE_RATE:.2f}s)  note={text_note}")

    received_events: list[dict] = []
    downlink_packets: list[bytes] = []
    saw_tts_start = saw_tts_stop = False
    first_audio_at: float | None = None
    sent_at: float | None = None

    async with websockets.connect(
        url,
        additional_headers={"device-id": DEVICE_ID, "protocol-version": "1"},
        max_size=None,
        open_timeout=15,
    ) as ws:
        await ws.send(json.dumps({
            "type": "hello",
            "version": 1,
            "transport": "websocket",
            "features": {"mcp": True},
            "audio_params": {"format": "opus", "sample_rate": SAMPLE_RATE,
                             "channels": 1, "frame_duration": 60},
        }))
        srv_hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
        print(f"[probe] server hello: {json.dumps(srv_hello)[:200]}")
        if srv_hello.get("transport") != "websocket":
            print("[probe] FATAL: server hello transport != websocket")
            return 2

        await ws.send(json.dumps({"type": "listen", "state": "start", "mode": "auto"}))
        print("[probe] sent listen:start")
        await asyncio.sleep(1.0)  # let the pipeline discard the wake-word tail

        async def reader():
            nonlocal saw_tts_start, saw_tts_stop, first_audio_at
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=idle_wait)
                    if isinstance(msg, bytes):
                        downlink_packets.append(msg)
                        if first_audio_at is None:
                            first_audio_at = time.monotonic()
                    else:
                        event = json.loads(msg)
                        received_events.append(event)
                        kind = event.get("type")
                        if kind == "tts" and event.get("state") == "start":
                            saw_tts_start = True
                        if kind == "tts" and event.get("state") == "stop":
                            saw_tts_stop = True
                        if kind in ("tts", "stt", "llm"):
                            print(f"[probe] <- {json.dumps(event)[:220]}")
            except asyncio.TimeoutError:
                print(f"[probe] reader idle timeout ({idle_wait}s) -> wrap up")
            except websockets.ConnectionClosed:
                print("[probe] connection closed by server")

        reader_task = asyncio.create_task(reader())
        await asyncio.sleep(0.3)

        sent_at = time.monotonic()
        for i in range(total_frames):
            chunk = pcm[i * frame_bytes:(i + 1) * frame_bytes]
            if len(chunk) < frame_bytes:
                break
            # Uplink must be Opus. Sending raw PCM makes the adapter log frames like
            # "latest=1920 bytes" and the decoder produces garbage -> fake endpoints.
            await ws.send(codec.encode(chunk))
            await asyncio.sleep(0.06)
        print(f"[probe] uploaded {total_frames} Opus frames")

        try:
            await asyncio.wait_for(reader_task, timeout=idle_wait + 30)
        except asyncio.TimeoutError:
            reader_task.cancel()
            await asyncio.gather(reader_task, return_exceptions=True)

    print("\n===== RESULT =====")
    print(f"tts:start={saw_tts_start}  tts:stop={saw_tts_stop}  "
          f"downlink audio packets={len(downlink_packets)}")
    if downlink_packets:
        sizes = [len(p) for p in downlink_packets]
        print(f"  packet bytes: min={min(sizes)} max={max(sizes)} total={sum(sizes)}")
        ttfa = (first_audio_at - sent_at) if first_audio_at and sent_at else None
        print(f"  TTFA(first downlink audio, from upload start) = {ttfa:.2f}s"
              if ttfa else "  TTFA=n/a")

    if downlink_packets:
        pcm_out = bytearray()
        for p in downlink_packets:
            try:
                pcm_out += codec.decode(p)
            except Exception as e:
                print(f"  decode fail: {type(e).__name__}: {e}")
        out_wav = out_dir / "e2e_downlink.wav"
        sf.write(out_wav, np.frombuffer(bytes(pcm_out), dtype="int16"), SAMPLE_RATE)
        print(f"[probe] downlink audio saved: {out_wav} "
              f"({len(pcm_out)/2/SAMPLE_RATE:.2f}s)")

        try:
            # Reuse production build_asr() so the feedback transcription uses exactly
            # the live ASR config (model / compute_type / language) instead of a guess.
            # Note: FasterWhisperASR's first arg is `model_name`, NOT `model_size`, and
            # transcribe() is a coroutine — forgetting `await` prints a <coroutine ...>.
            from app.main import build_asr

            text = await build_asr().transcribe(bytes(pcm_out))
            print(f"[probe] ASR feedback = {text!r}")
        except Exception as e:
            print(f"[probe] ASR feedback failed: {type(e).__name__}: {e}")

    for ev in received_events:
        if ev.get("type") == "stt":
            print(f"[probe] ASR event = {json.dumps(ev)[:300]}")
    codec.close()
    return 0 if downlink_packets else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav", help="16k mono wav to feed as device uplink")
    src.add_argument("--say", help="generate the uplink wav from this text via macOS say")
    ap.add_argument("--voice", help="say voice name (used with --say), e.g. Meijia for Chinese")
    ap.add_argument("--url", default="ws://127.0.0.1:8003/xiaozhi/ws")
    ap.add_argument("--text", default="", help="free-form note printed with the run")
    ap.add_argument("--idle-wait", type=float, default=25.0)
    ap.add_argument("--out-dir", default="/tmp/linkdog_e2e")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.say:
        wav = out_dir / "probe_say.wav"
        _build_say_wav(a.say, wav, voice=a.voice)
        wav_path = str(wav)
    else:
        wav_path = a.wav

    return asyncio.run(
        run(a.url, load_pcm_16k_mono(wav_path), a.text, a.idle_wait, out_dir)
    )


if __name__ == "__main__":
    raise SystemExit(main())
