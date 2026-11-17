import unittest

from scripts.compaction_watermarks import calculate_watermark_chars


class CalculateWatermarkCharsTests(unittest.TestCase):
    def test_default_watermarks_scale_with_context(self):
        at_32k = calculate_watermark_chars(32768)
        at_64k = calculate_watermark_chars(65536)

        self.assertEqual(at_32k, {"soft": 73728, "high": 92160, "low": 49152})
        self.assertEqual(at_64k, {key: value * 2 for key, value in at_32k.items()})

    def test_custom_char_ratio_and_watermark_ratios(self):
        limits = calculate_watermark_chars(
            10000,
            chars_per_token=4,
            soft_ratio=0.5,
            high_ratio=0.8,
            low_ratio=0.25,
        )
        self.assertEqual(limits, {"soft": 20000, "high": 32000, "low": 10000})

    def test_rejects_invalid_context_or_ratio_order(self):
        with self.assertRaises(ValueError):
            calculate_watermark_chars(0)
        with self.assertRaises(ValueError):
            calculate_watermark_chars(32768, low_ratio=0.8)


if __name__ == "__main__":
    unittest.main()
