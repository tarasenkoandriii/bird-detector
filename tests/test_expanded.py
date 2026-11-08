"""Нові holdout не можуть містити вже навчальні зображення."""

import unittest
from prepare_expanded import choose_splits


class ExpandedSplits(unittest.TestCase):
    def test_previous_split_protection(self):
        records = {
            "train": [(x, {}) for x in ["trained", "oldval", "oldtest", "new1", "new2", "new3"]],
            "test": [(x, {}) for x in ["oldtest", "trained", "oldval", "fresh1", "fresh2"]],
        }
        parts, newval = choose_splits(records, {"trained"}, {"oldval"}, {"oldtest"}, 2, 2)
        ids = {k: {h for h, _ in v} for k, v in parts.items()}
        self.assertIn("trained", ids["train"])
        self.assertIn("oldval", ids["val"])
        self.assertEqual(ids["test"], {"fresh1", "fresh2"})
        self.assertNotIn("oldtest", set.union(*ids.values()))
        self.assertFalse(ids["train"] & ids["val"])
        self.assertFalse(newval & {"trained", "oldval", "oldtest"})
        self.assertEqual(
            parts, choose_splits(records, {"trained"}, {"oldval"}, {"oldtest"}, 2, 2)[0]
        )


if __name__ == "__main__":
    unittest.main()
