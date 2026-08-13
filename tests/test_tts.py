import unittest
from unittest.mock import patch

from voice.tts import TextToSpeech


class TestTTS(unittest.TestCase):

    def test_validate_text_empty(self):
        tts = TextToSpeech()

        with self.assertRaises(ValueError):
            tts._validate_text("")


if __name__ == "__main__":
    unittest.main()