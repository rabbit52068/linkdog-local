import unittest

import numpy as np

from app.pocket_tts import PocketTTSBackend
from app.tts import TTSError


class FakePocketModel:
    def __init__(self):
        self.state_calls = []
        self.generate_calls = []

    def get_state_for_audio_prompt(self, voice):
        self.state_calls.append(voice)
        return {"voice": voice}

    def generate_audio_stream(self, state, text, copy_state=True):
        self.generate_calls.append((state, text, copy_state))
        yield np.full(1200, 0.5, dtype=np.float32)
        yield np.full(1200, -0.5, dtype=np.float32)


class PocketTTSBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_loads_model_and_voice_once_and_returns_16khz_s16le(self):
        model = FakePocketModel()
        factory_calls = []

        def factory():
            factory_calls.append(True)
            return model

        backend = PocketTTSBackend(voice="cosette", model_factory=factory)

        first = await backend.synthesize("Hello, Nelson!")
        second = await backend.synthesize("Sit down.")

        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(model.state_calls, ["cosette"])
        self.assertEqual(len(model.generate_calls), 2)
        self.assertEqual(len(first), 1600 * 2)
        self.assertEqual(len(second), 1600 * 2)
        self.assertEqual(len(first) % 2, 0)
        samples = np.frombuffer(first, dtype="<i2")
        self.assertGreater(samples.max(), 15000)
        self.assertLess(samples.min(), -15000)

    async def test_rejects_blank_text(self):
        backend = PocketTTSBackend(model_factory=FakePocketModel)

        with self.assertRaisesRegex(TTSError, "text is blank"):
            await backend.synthesize("   ")


if __name__ == "__main__":
    unittest.main()
