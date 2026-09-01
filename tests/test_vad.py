import unittest

from app.vad import UtteranceEndpoint, VadConfigurationError


CHUNK_BYTES = 640  # 20 ms, 16 kHz, mono, signed 16-bit PCM


def chunk(marker: int) -> bytes:
    return bytes([marker]) * CHUNK_BYTES


class MarkerVad:
    def is_speech(self, pcm: bytes) -> bool:
        return pcm[0] == 1


class UtteranceEndpointTests(unittest.TestCase):
    def endpoint(self, **overrides):
        settings = {
            "classifier": MarkerVad(),
            "sample_rate": 16_000,
            "chunk_duration_ms": 20,
            "pre_roll_ms": 40,
            "minimum_speech_ms": 40,
            "end_silence_ms": 40,
            "maximum_utterance_ms": 400,
        }
        settings.update(overrides)
        return UtteranceEndpoint(**settings)

    def test_returns_utterance_with_pre_roll_after_trailing_silence(self):
        endpoint = self.endpoint()

        self.assertIsNone(endpoint.process_pcm(chunk(0)))
        self.assertIsNone(endpoint.process_pcm(chunk(0)))
        self.assertIsNone(endpoint.process_pcm(chunk(1)))
        self.assertIsNone(endpoint.process_pcm(chunk(1)))
        self.assertIsNone(endpoint.process_pcm(chunk(0)))
        utterance = endpoint.process_pcm(chunk(0))

        self.assertEqual(
            utterance,
            chunk(0) + chunk(1) + chunk(1) + chunk(0) + chunk(0),
        )
        self.assertFalse(endpoint.speech_active)

    def test_discards_burst_shorter_than_minimum_speech(self):
        endpoint = self.endpoint(minimum_speech_ms=60)

        endpoint.process_pcm(chunk(1))
        endpoint.process_pcm(chunk(0))
        self.assertIsNone(endpoint.process_pcm(chunk(0)))
        self.assertFalse(endpoint.speech_active)

    def test_maximum_duration_forces_endpoint_without_silence(self):
        endpoint = self.endpoint(
            pre_roll_ms=0,
            minimum_speech_ms=20,
            maximum_utterance_ms=60,
        )

        self.assertIsNone(endpoint.process_pcm(chunk(1)))
        self.assertIsNone(endpoint.process_pcm(chunk(1)))
        utterance = endpoint.process_pcm(chunk(1))

        self.assertEqual(utterance, chunk(1) * 3)

    def test_reset_discards_previous_audio(self):
        endpoint = self.endpoint()
        endpoint.process_pcm(chunk(0))
        endpoint.process_pcm(chunk(1))

        endpoint.reset()
        endpoint.process_pcm(chunk(1))
        endpoint.process_pcm(chunk(1))
        endpoint.process_pcm(chunk(0))
        utterance = endpoint.process_pcm(chunk(0))

        self.assertEqual(utterance, chunk(1) + chunk(1) + chunk(0) + chunk(0))

    def test_rejects_pcm_not_aligned_to_vad_chunks(self):
        endpoint = self.endpoint()

        with self.assertRaisesRegex(ValueError, "whole 20 ms chunks"):
            endpoint.process_pcm(b"bad")

    def test_rejects_threshold_not_aligned_to_chunk_duration(self):
        with self.assertRaises(VadConfigurationError):
            self.endpoint(end_silence_ms=30)


if __name__ == "__main__":
    unittest.main()
