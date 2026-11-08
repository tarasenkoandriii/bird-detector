"""Середнє не повинно приховувати невдалі спільні порівняння."""

import unittest
from seed_study import aggregate, stats


class SeedStudy(unittest.TestCase):
    def test_average_and_joint_passes_are_separate(self):
        rows = [
            {"cub": 0.81, "video": 0.52, "background_fp_images": 0},
            {"cub": 0.79, "video": 0.53, "background_fp_images": 0},
            {"cub": 0.82, "video": 0.49, "background_fp_images": 0},
        ]
        s = aggregate(rows, {"cub": 0.8, "video": 0.5, "background_fp_images": 0})
        self.assertAlmostEqual(s["cub"]["mean"], 0.8066666666666666)
        self.assertAlmostEqual(s["cub"]["sd"], 0.0152752523165195)
        self.assertEqual(s["joint_passes"], 1)

    def test_exact_tie_is_not_an_improvement(self):
        s = aggregate(
            [{"cub": 0.8, "video": 0.5, "background_fp_images": 0}],
            {"cub": 0.8, "video": 0.5, "background_fp_images": 0},
        )
        self.assertEqual(s["joint_passes"], 0)

    def test_single_seed_has_no_estimated_spread(self):
        self.assertIsNone(stats([0.8])["sd"])
        self.assertEqual(stats([0.8, 0.8, 0.8])["sd"], 0)

    def test_background_regression_fails_joint_criterion(self):
        s = aggregate(
            [{"cub": 0.9, "video": 0.9, "background_fp_images": 1}],
            {"cub": 0.8, "video": 0.5, "background_fp_images": 0},
        )
        self.assertEqual(s["joint_passes"], 0)


if __name__ == "__main__":
    unittest.main()
