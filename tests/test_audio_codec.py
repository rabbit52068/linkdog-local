import math
import struct
import unittest

from app.audio_codec import OpusCodec, OpusCodecError


SAMPLE_RATE = 16_000
CHANNELS = 1
FRAME_DURATION_MS = 60
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_DURATION_MS // 1000


def sine_frame(frequency: float = 440.0, amplitude: int = 10_000) -> bytes:
    samples = [
        int(amplitude * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE))
        for index in range(SAMPLES_PER_FRAME)
    ]
    return struct.pack(f"<{SAMPLES_PER_FRAME}h", *samples)


class OpusCodecTests(unittest.TestCase):
    def setUp(self):
        self.codec = OpusCodec(
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            frame_duration_ms=FRAME_DURATION_MS,
        )

    def tearDown(self):
        self.codec.close()

    def test_round_trip_preserves_frame_shape(self):
        pcm = sine_frame()

        packet = self.codec.encode(pcm)
        decoded = self.codec.decode(packet)

        self.assertIsInstance(packet, bytes)
        self.assertGreater(len(packet), 0)
        self.assertLess(len(packet), len(pcm))
        self.assertEqual(len(decoded), len(pcm))
        self.assertEqual(len(decoded) // 2, SAMPLES_PER_FRAME)

    def test_encode_rejects_pcm_with_wrong_frame_size(self):
        with self.assertRaisesRegex(OpusCodecError, "exactly 960 samples"):
            self.codec.encode(b"\x00\x00" * (SAMPLES_PER_FRAME - 1))

    def test_decode_rejects_empty_packet(self):
        with self.assertRaisesRegex(OpusCodecError, "empty Opus packet"):
            self.codec.decode(b"")

    def test_close_is_idempotent(self):
        self.codec.close()
        self.codec.close()


if __name__ == "__main__":
    unittest.main()
