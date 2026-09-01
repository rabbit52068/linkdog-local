import asyncio
import struct
import threading
import unittest

from app.asr import ASRError, ASRTimeoutError, FasterWhisperASR


PCM_60MS = struct.pack("<960h", *([1_000] * 960))


class Segment:
    def __init__(self, text):
        self.text = text


class RecordingModel:
    def __init__(self, segments=None, error=None):
        self.segments = segments or []
        self.error = error
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        if self.error is not None:
            raise self.error
        return iter(self.segments), object()


class FasterWhisperASRTests(unittest.IsolatedAsyncioTestCase):
    async def test_lazy_loads_model_once_and_normalizes_pcm(self):
        model = RecordingModel([Segment(" 你好"), Segment("世界 ")])
        factory_calls = []

        def factory(model_name, device, compute_type):
            factory_calls.append((model_name, device, compute_type))
            return model

        backend = FasterWhisperASR(
            model_name="base",
            device="cpu",
            compute_type="int8",
            language="zh",
            model_factory=factory,
            timeout_seconds=1,
        )

        first = await backend.transcribe(PCM_60MS, sample_rate=16_000)
        second = await backend.transcribe(PCM_60MS, sample_rate=16_000)

        self.assertEqual(first, "你好 世界")
        self.assertEqual(second, "你好 世界")
        self.assertEqual(factory_calls, [("base", "cpu", "int8")])
        audio, kwargs = model.calls[0]
        self.assertEqual(audio.dtype.name, "float32")
        self.assertEqual(audio.shape, (960,))
        self.assertAlmostEqual(float(audio[0]), 1_000 / 32_768, places=6)
        self.assertEqual(kwargs["language"], "zh")
        self.assertEqual(kwargs["beam_size"], 5)
        self.assertFalse(kwargs["vad_filter"])
        self.assertEqual(kwargs["no_speech_threshold"], 0.6)
        self.assertIsNone(kwargs["log_prob_threshold"])

    async def test_returns_empty_string_when_model_has_no_text(self):
        backend = FasterWhisperASR(
            model_factory=lambda *_args: RecordingModel([Segment("   ")]),
            timeout_seconds=1,
        )

        result = await backend.transcribe(PCM_60MS, sample_rate=16_000)

        self.assertEqual(result, "")

    async def test_rejects_invalid_pcm_before_loading_model(self):
        loaded = False

        def factory(*_args):
            nonlocal loaded
            loaded = True
            return RecordingModel()

        backend = FasterWhisperASR(model_factory=factory)

        with self.assertRaisesRegex(ASRError, "16 kHz"):
            await backend.transcribe(PCM_60MS, sample_rate=8_000)
        with self.assertRaisesRegex(ASRError, "non-empty"):
            await backend.transcribe(b"", sample_rate=16_000)
        with self.assertRaisesRegex(ASRError, "16-bit"):
            await backend.transcribe(b"x", sample_rate=16_000)
        self.assertFalse(loaded)

    async def test_wraps_model_failure(self):
        backend = FasterWhisperASR(
            model_factory=lambda *_args: RecordingModel(error=RuntimeError("broken")),
            timeout_seconds=1,
        )

        with self.assertRaisesRegex(ASRError, "transcription failed"):
            await backend.transcribe(PCM_60MS, sample_rate=16_000)

    async def test_times_out_without_blocking_event_loop(self):
        release = threading.Event()

        class BlockingModel:
            def transcribe(self, _audio, **_kwargs):
                release.wait(timeout=1)
                return iter([]), object()

        backend = FasterWhisperASR(
            model_factory=lambda *_args: BlockingModel(),
            timeout_seconds=0.01,
        )
        try:
            with self.assertRaises(ASRTimeoutError):
                await backend.transcribe(PCM_60MS, sample_rate=16_000)
        finally:
            release.set()
            await asyncio.sleep(0.01)


if __name__ == "__main__":
    unittest.main()
