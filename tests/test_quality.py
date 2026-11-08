"""Геометрія flip-fusion без залежності від конкретних ваг."""

import unittest
import numpy as np
from quality import unflip_boxes, fuse_two_views


class Fusion(unittest.TestCase):
    def test_flip_twice_restores_coordinates_and_score(self):
        boxes = [[10, 5, 30, 25, 0.8]]
        self.assertTrue(np.allclose(unflip_boxes(unflip_boxes(boxes, 100), 100), boxes))
        self.assertTrue(np.allclose(unflip_boxes(boxes, 100), [[70, 5, 90, 25, 0.8]]))

    def test_paired_box_is_weighted_and_confidence_not_inflated(self):
        out = fuse_two_views([[0, 0, 10, 10, 0.8]], [[2, 0, 12, 10, 0.6]])
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0][0], 1.2 / 1.4)
        self.assertAlmostEqual(out[0][4], 0.7)

    def test_empty_and_unconfirmed_separate_birds(self):
        self.assertEqual(fuse_two_views([], []), [])
        out = fuse_two_views([[0, 0, 10, 10, 0.8]], [[50, 0, 60, 10, 0.6]])
        self.assertEqual(len(out), 2)
        self.assertTrue(np.allclose(sorted(x[4] for x in out), [0.3, 0.4]))


if __name__ == "__main__":
    unittest.main()
