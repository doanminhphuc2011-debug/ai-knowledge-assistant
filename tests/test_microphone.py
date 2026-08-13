import unittest
from unittest.mock import patch
import numpy as np

from voice.microphone import Microphone


class TestMicrophone(unittest.TestCase):

    @patch("voice.microphone.sd.wait")
    @patch("voice.microphone.sd.rec")
    def test_record_returns_1d_array(
        self,
        mock_rec,
        mock_wait,
    ):
        mock_rec.return_value = np.zeros(
            (16000, 1),
            dtype=np.float32,
        )

        mic = Microphone()

        audio = mic.record(1)

        self.assertEqual(audio.ndim, 1)


if __name__ == "__main__":
    unittest.main()