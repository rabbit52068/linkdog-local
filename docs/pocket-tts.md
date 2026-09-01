# Pocket TTS integration

Local neural text-to-speech for the LinkDog adapter, replacing the cloud `edge-tts` / macOS `say` fallback.

## Why

Pocket TTS runs fully offline with a built-in voice, so the dog keeps speaking without an internet connection or a cloud TTS API key.

## Dependencies

```bash
uv venv --python 3.11 <venv>
uv pip install --python <venv>/bin/python pocket-tts scipy soundfile
```

- `pocket-tts` (3.x)
- `torch` (CPU wheel is fine on macOS)
- `scipy`, `soundfile`

Built-in voices do not require a Hugging Face login or gated access (only voice cloning does).

## Interface contract

The TTS backend must match the existing `CommandTTSBackend` contract in `app/tts.py`:

```python
async def synthesize(self, text: str) -> bytes
# returns 16 kHz, mono, s16le PCM bytes
```

Downstream, `OpusDownlinkPlayer` encodes with `OpusCodec(sample_rate=16_000, channels=1)`.

## Reference implementation (`app/pocket_tts.py`)

```python
import asyncio
import numpy as np
import torch
from scipy.signal import resample_poly
from pocket_tts import TTSModel


class PocketTTSBackend:
    """Resident Pocket TTS (built-in voice) → 16 kHz mono s16le PCM."""

    def __init__(self, voice: str = "cosette"):
        self.voice = voice
        self._model = None
        self._state = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model = TTSModel.load_model()          # resident, load once
            self._state = self._model.get_state_for_audio_prompt(self.voice)

    async def synthesize(self, text: str) -> bytes:
        text = text.strip()
        if not text:
            raise RuntimeError("TTS text is blank")
        return await asyncio.to_thread(self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> bytes:
        self._ensure_loaded()
        chunks = []
        for chunk in self._model.generate_audio_stream(
            self._state, text, copy_state=True
        ):
            if isinstance(chunk, torch.Tensor):
                chunk = chunk.detach().cpu().float().numpy()
            chunks.append(np.asarray(chunk).reshape(-1))
        audio = np.concatenate(chunks)                    # 24 kHz float32 mono

        pcm16k = resample_poly(audio, 2, 3)               # 24k → 16k
        pcm16k = np.clip(pcm16k, -1.0, 1.0)
        s16 = (pcm16k * 32767.0).astype(np.int16)
        return s16.tobytes()
```

## Wiring into `build_tts()` (`app/main.py`)

```python
def build_tts():
    if os.environ.get("LINKDOG_TTS_BACKEND", "").strip().lower() == "pocket":
        from app.pocket_tts import PocketTTSBackend
        return PocketTTSBackend(
            voice=os.environ.get("LINKDOG_POCKET_VOICE", "cosette")
        )
    # existing CommandTTSBackend logic unchanged
    ...
```

`.env`:

```text
LINKDOG_TTS_BACKEND=pocket
LINKDOG_POCKET_VOICE=cosette
```

## Notes

1. Keep the model resident — call `load_model()` once at startup.
2. Build the voice state once and reuse it.
3. Resample in-process with `scipy.signal.resample_poly`; do not spawn ffmpeg per sentence.
4. `torch` is a heavy dependency; the CPU wheel is sufficient on macOS.
5. Resident RAM is roughly 0.8 GB.
6. Built-in voices need no HF login; voice cloning requires gated access.

## Known limitations

- Built-in voice character (cuteness / youth) must be judged by ear; `cosette` is the livelier of the built-in options.
- Cloning a custom voice requires a clean reference WAV and HF gated access.
