import asyncio
import subprocess
import unittest

from app.tts import (
    CommandTTSBackend,
    TTSError,
    extract_emotion_prefix,
    map_emotion,
    pcm_frames,
    sanitize_spoken_text,
)


class FakeRunner:
    def __init__(self, edge_returncode=0, say_returncode=0, ffmpeg_pcm=b"\x01\x00" * 960):
        self.edge_returncode = edge_returncode
        self.say_returncode = say_returncode
        self.ffmpeg_pcm = ffmpeg_pcm
        self.commands = []

    def __call__(self, args, **_kwargs):
        self.commands.append(args)
        executable = args[0]
        if executable == "edge-tts":
            return subprocess.CompletedProcess(args, self.edge_returncode, b"", b"edge failed")
        if executable == "say":
            return subprocess.CompletedProcess(args, self.say_returncode, b"", b"say failed")
        if executable == "ffmpeg":
            return subprocess.CompletedProcess(args, 0, self.ffmpeg_pcm, b"")
        raise AssertionError(f"unexpected executable: {executable}")


class CommandTTSBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_macos_say_audio_is_normalized_to_16khz_mono_s16le(self):
        runner = FakeRunner(ffmpeg_pcm=b"\x02\x00" * 960)
        backend = CommandTTSBackend(
            edge_tts_command="edge-tts",
            ffmpeg_command="ffmpeg",
            say_command="say",
            runner=runner,
        )

        pcm = await backend.synthesize("你好")

        self.assertEqual(pcm, b"\x02\x00" * 960)
        self.assertEqual(runner.commands[0][0], "say")
        self.assertIn("Meijia", runner.commands[0])
        self.assertEqual(runner.commands[1][0], "ffmpeg")
        self.assertIn("16000", runner.commands[1])
        self.assertIn("s16le", runner.commands[1])

    async def test_edge_tts_is_used_when_macos_say_fails(self):
        runner = FakeRunner(say_returncode=1)
        backend = CommandTTSBackend(runner=runner)

        pcm = await backend.synthesize("連線失敗")

        self.assertTrue(pcm)
        self.assertEqual([command[0] for command in runner.commands], [
            "say",
            "edge-tts",
            "ffmpeg",
        ])

    async def test_raises_when_both_tts_providers_fail(self):
        runner = FakeRunner(edge_returncode=1, say_returncode=1)
        backend = CommandTTSBackend(runner=runner)

        with self.assertRaisesRegex(TTSError, "macOS say and Edge TTS failed"):
            await backend.synthesize("失敗")

    async def test_rejects_blank_text(self):
        backend = CommandTTSBackend(runner=FakeRunner())

        with self.assertRaisesRegex(TTSError, "text is blank"):
            await backend.synthesize("   ")


class PCMFrameTests(unittest.TestCase):
    def test_splits_pcm_into_60ms_frames_and_pads_final_frame(self):
        frame_bytes = 960 * 2
        pcm = b"\x01\x00" * (960 + 100)

        frames = list(pcm_frames(pcm, samples_per_frame=960, channels=1))

        self.assertEqual(len(frames), 2)
        self.assertEqual(len(frames[0]), frame_bytes)
        self.assertEqual(len(frames[1]), frame_bytes)
        self.assertEqual(frames[1][:200], b"\x01\x00" * 100)
        self.assertEqual(frames[1][200:], b"\x00" * (frame_bytes - 200))

    def test_rejects_odd_length_pcm(self):
        with self.assertRaisesRegex(TTSError, "16-bit aligned"):
            list(pcm_frames(b"\x00", samples_per_frame=960, channels=1))

    def test_rejects_empty_pcm(self):
        with self.assertRaisesRegex(TTSError, "PCM audio is empty"):
            list(pcm_frames(b"", samples_per_frame=960, channels=1))


class EmotionTests(unittest.TestCase):
    def test_extracts_whitelisted_emotion_prefix_from_spoken_text(self):
        self.assertEqual(
            extract_emotion_prefix("😘 I missed you!"),
            ("I missed you!", "kissy"),
        )

    def test_text_without_emotion_prefix_is_unchanged(self):
        self.assertEqual(
            extract_emotion_prefix("Hello there!"),
            ("Hello there!", None),
        )

    def test_maps_known_tones_deterministically(self):
        self.assertEqual(map_emotion("That's great!"), "happy")
        self.assertEqual(map_emotion("Sorry, the connection failed."), "sad")
        self.assertEqual(map_emotion("What do you want to ask?"), "thinking")

    def test_unknown_tone_defaults_to_neutral(self):
        self.assertEqual(map_emotion("The temperature is twenty-five degrees."), "neutral")


class SpokenTextSafetyTests(unittest.TestCase):
    def test_wake_word_is_not_spoken_back_through_device_speaker(self):
        self.assertEqual(
            sanitize_spoken_text("請說小斌小斌來喚醒我"),
            "請說喚醒詞來喚醒我",
        )


if __name__ == "__main__":
    unittest.main()
