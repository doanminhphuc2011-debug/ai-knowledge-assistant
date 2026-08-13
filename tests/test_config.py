import unittest
from voice.config import get_voice_config


class TestConfig(unittest.TestCase):

    def test_load_config(self):
        config = get_voice_config()

        self.assertTrue(config.enabled)
        self.assertEqual(config.language, "vi")
        self.assertEqual(config.sample_rate, 16000)


if __name__ == "__main__":
    unittest.main()