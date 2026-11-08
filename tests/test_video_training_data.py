"""Захист відеотесту, негативних вирізок і масштабу дрібних цілей."""

import unittest
from prepare_video_training import validate_selection, crop_clear, small_target


class VideoTrainingData(unittest.TestCase):
    def test_reserved_video_is_rejected(self):
        for seq in ["bird-2", "bird-3", "bird-15", "bird-17"]:
            with self.assertRaises(ValueError):
                validate_selection(
                    {
                        "reviewed": True,
                        "records": [{"sequence": seq, "split": "train", "kind": "positive"}],
                    }
                )
        with self.assertRaises(ValueError):
            validate_selection({"reviewed": False, "records": []})

    def test_negative_crop_requires_margin(self):
        self.assertTrue(crop_clear([0, 0, 100, 100], [150, 150, 80, 80]))
        self.assertFalse(crop_clear([100, 100, 200, 200], [150, 150, 80, 80]))

    def test_small_target_uses_letterbox_scale(self):
        self.assertTrue(small_target([0, 0, 25, 50], [1280, 720]))
        self.assertFalse(small_target([0, 0, 50, 50], [480, 360]))


if __name__ == "__main__":
    unittest.main()
