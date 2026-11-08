"""Цільовий recall не повинен включати невидимих птахів або вигадувати FP."""

import unittest
from evaluate_video_test import target_metrics


class TargetMetrics(unittest.TestCase):
    def test_visibility_and_misses(self):
        rows = [
            {"visible": True, "best_iou": 0.8, "detections": [[1]]},
            {"visible": True, "best_iou": 0, "detections": []},
            {"visible": False, "best_iou": 0, "detections": []},
        ]
        m = target_metrics(rows)
        self.assertEqual(m["target_recall_iou50"], 0.5)
        self.assertEqual(m["mean_best_iou_visible"], 0.4)
        self.assertEqual(m["no_detections_visible"], 1)
        self.assertEqual(m["excluded_invisible_target_frames"], 1)

    def test_no_visible_targets(self):
        self.assertIsNone(target_metrics([])["target_recall_iou50"])


if __name__ == "__main__":
    unittest.main()
