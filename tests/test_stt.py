import unittest
import numpy as np

from voice.stt import SpeechToText


class TestSTTValidation(unittest.TestCase):

    def test_validate_audio_not_ndarray(self):
        with self.assertRaises(ValueError):
            SpeechToText._validate_audio("abc")

    def test_validate_audio_empty(self):
        with self.assertRaises(ValueError):
            SpeechToText._validate_audio(np.array([], dtype=np.float32))

    def test_validate_audio_not_1d(self):
        with self.assertRaises(ValueError):
            SpeechToText._validate_audio(
                np.zeros((2, 2), dtype=np.float32)
            )


if __name__ == "__main__":
    unittest.main()