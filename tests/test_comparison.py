"""Регресія: COCO person=0 не має помилково враховуватися як bird=0."""

import unittest
from types import SimpleNamespace
import torch
from compare_models import bird_validator


class Mapping(unittest.TestCase):
    def test_filter_original_bird_then_map_to_zero(self):
        validator = object.__new__(bird_validator(14))
        validator.args = SimpleNamespace(conf=0.001, iou=0.7, max_det=300)
        validator.end2end = True
        raw = torch.tensor(
            [
                [
                    [0.0, 0.0, 10.0, 10.0, 0.99, 0.0],
                    [20.0, 20.0, 40.0, 40.0, 0.9, 14.0],
                    [50.0, 50.0, 80.0, 80.0, 0.8, 2.0],
                ]
            ]
        )
        out = validator.postprocess(raw)[0]
        self.assertEqual(len(out["bboxes"]), 1)
        self.assertEqual(out["cls"].tolist(), [0.0])
        self.assertEqual(out["bboxes"].tolist(), [[20.0, 20.0, 40.0, 40.0]])


class FreezeSelection(unittest.TestCase):
    def test_only_classifier_in_last_head_is_trainable(self):
        from train_head import classifier_parameter

        self.assertTrue(classifier_parameter("model.23.cv3.0.2.weight", 23))
        self.assertTrue(classifier_parameter("model.23.one2one_cv3.0.2.weight", 23))
        self.assertFalse(classifier_parameter("model.23.cv2.0.0.conv.weight", 23))
        self.assertFalse(classifier_parameter("model.23.one2one_cv2.0.0.conv.weight", 23))
        self.assertFalse(classifier_parameter("model.4.cv3.conv.weight", 23))


class BoxFreezeSelection(unittest.TestCase):
    def test_bbox_only_in_last_head(self):
        from train_boxes import box_parameter

        self.assertTrue(box_parameter("model.23.cv2.0.0.conv.weight", 23))
        self.assertTrue(box_parameter("model.23.one2one_cv2.0.0.conv.weight", 23))
        self.assertFalse(box_parameter("model.23.cv3.0.2.weight", 23))
        self.assertFalse(box_parameter("model.4.cv2.conv.weight", 23))


class Matching(unittest.TestCase):
    def test_duplicates_and_disjoint_predictions(self):
        from analyze_errors import match

        tp, fp, fn, _ = match([[0, 0, 10, 10]], [[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]])
        self.assertEqual((tp, fp, fn), (1, 2, 0))

    def test_missing_prediction(self):
        from analyze_errors import match

        self.assertEqual(match([[0, 0, 10, 10]], [])[:3], (0, 0, 1))


if __name__ == "__main__":
    unittest.main()
