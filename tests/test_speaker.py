import unittest
from voice.speaker import Speaker


class TestSpeaker(unittest.TestCase):

    def test_validate_audio_empty(self):
        with self.assertRaises(ValueError):
            Speaker._validate_audio(b"")

    def test_validate_audio_wrong_type(self):
        with self.assertRaises(ValueError):
            Speaker._validate_audio("abc")


if __name__ == "__main__":
    unittest.main()