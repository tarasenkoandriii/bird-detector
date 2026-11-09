import unittest

import numpy as np

from evaluate_tiled import axis_windows, tile_windows, restore_boxes, merge_boxes, detection_metrics


class TiledGeometryTests(unittest.TestCase):
    def test_edges_covered_without_empty_or_duplicate_windows(self):
        windows = axis_windows(1280, 640, .20)
        self.assertEqual(windows, [(0, 640), (512, 1152), (640, 1280)])
        self.assertEqual(len(windows), len(set(windows)))
        covered = np.zeros(1280, dtype=bool)
        for start, end in windows:
            covered[start:end] = True
        self.assertTrue(covered.all())

    def test_smaller_edge_image_is_a_single_partial_tile(self):
        self.assertEqual(tile_windows(319, 217), [(0, 0, 319, 217)])

    def test_offset_clipping_and_degenerate_boxes(self):
        original = np.array([[1, 2, 101, 82, .9, 0], [5, 5, 5, 8, .5, 0]], np.float32)
        restored = restore_boxes(original, (600, 650), (640, 720))
        np.testing.assert_allclose(restored, [[601, 652, 640, 720, .9, 0]], rtol=1e-6)
        self.assertEqual(original[0, 0], 1)  # не мутує вхід

    def test_cross_tile_duplicates_removed_and_scores_sorted(self):
        boxes = [[100, 100, 200, 200, .6, 0], [102, 102, 202, 202, .9, 0],
                 [300, 300, 350, 350, .8, 0]]
        merged = merge_boxes(boxes)
        self.assertEqual(len(merged), 2)
        np.testing.assert_allclose(merged[:, 4], [.9, .8])

    def test_ap_matching_uses_ground_truth_rows_and_prediction_columns(self):
        perfect = [dict(targets=[[10, 20, 50, 80]], predictions=[[10, 20, 50, 80, .9, 0]])]
        metrics = detection_metrics(perfect)
        self.assertGreater(metrics['map50_95'], .99)
        wrong_first = [dict(targets=[[10, 20, 50, 80]], predictions=[
            [100, 100, 140, 160, .95, 0], [10, 20, 50, 80, .9, 0]])]
        self.assertLess(detection_metrics(wrong_first)['map50'], metrics['map50'])


if __name__ == '__main__':
    unittest.main()
