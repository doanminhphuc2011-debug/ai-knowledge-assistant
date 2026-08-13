import unittest

from voice.text_normalizer import TextNormalizer


class TestTextNormalizer(unittest.TestCase):

    def setUp(self):
        self.normalizer = TextNormalizer()

    def test_whitespace(self):
        self.assertEqual(
            self.normalizer.normalize("   xin    chào   "),
            "xin chào",
        )

    def test_uppercase(self):
        self.assertEqual(
            self.normalizer.normalize("XIN CHÀO"),
            "xin chào",
        )

    def test_filler_word(self):
        self.assertEqual(
            self.normalizer.normalize("à tôi muốn gọi cà phê"),
            "tôi muốn gọi cà phê",
        )


if __name__ == "__main__":
    unittest.main()