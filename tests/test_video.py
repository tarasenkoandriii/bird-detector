import argparse
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import cv2
import torch
import predict_video


class FakeModel:
    names = {0: "bird"}

    def __init__(self, _):
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1
        boxes = (
            torch.tensor([[5.0, 5.0, 45.0, 45.0, 0.8, 0.0]])
            if self.calls % 2
            else torch.empty((0, 6))
        )
        return [SimpleNamespace(boxes=SimpleNamespace(data=boxes))]


class FailingModel(FakeModel):
    def predict(self, **kwargs):
        if self.calls == 1:
            raise RuntimeError("змодельований збій інференсу")
        return super().predict(**kwargs)


class VideoTests(unittest.TestCase):
    def fixture(self, root):
        import numpy as np

        source = root / "in.mp4"
        weights = root / "best.pt"
        weights.write_bytes(b"fake")
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 12.5, (96, 64))
        self.assertTrue(writer.isOpened())
        for _ in range(4):
            writer.write(np.full((64, 96, 3), 110, dtype=np.uint8))
        writer.release()
        return argparse.Namespace(
            input=source,
            output=root / "out.mp4",
            weights=weights,
            conf=0.25,
            iou=0.5,
            imgsz=320,
            device="cpu",
            overwrite=False,
        )

    def test_all_frames_and_background_frames_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.fixture(Path(d))
            report = predict_video.run(a, FakeModel)
            self.assertEqual(report["frames"], 4)
            self.assertEqual(report["output_frames"], 4)
            self.assertEqual(report["frames_with_detections"], 2)
            self.assertEqual(report["fps"], 12.5)
            self.assertEqual(report["skipped_frames"], 0)
            self.assertTrue(a.output.is_file())
            self.assertEqual(list(Path(d).glob(".bird-*")), [])

    def test_failure_does_not_publish_partial_output(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.fixture(Path(d))
            before = hashlib.sha256(a.input.read_bytes()).hexdigest()
            with self.assertRaises(RuntimeError):
                predict_video.run(a, FailingModel)
            self.assertFalse(a.output.exists())
            self.assertEqual(list(Path(d).glob(".bird-*")), [])
            self.assertEqual(before, hashlib.sha256(a.input.read_bytes()).hexdigest())

    def test_refuses_overwrite_and_same_input(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.fixture(Path(d))
            a.output.write_bytes(b"keep")
            with self.assertRaises(ValueError):
                predict_video.run(a, FakeModel)
            self.assertEqual(a.output.read_bytes(), b"keep")
            a.output = a.input
            with self.assertRaises(ValueError):
                predict_video.validate(a)

    def test_suppresses_overlapping_copy_but_keeps_separate_bird(self):
        boxes = [[0, 0, 40, 40, 0.9], [1, 1, 41, 41, 0.7], [60, 0, 90, 40, 0.8]]
        result = predict_video.suppress_duplicates(boxes, 0.5)
        self.assertEqual(len(result), 2)
        self.assertIn(boxes[0], result)
        self.assertIn(boxes[2], result)

    def test_missing_or_corrupt_input(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.fixture(Path(d))
            a.input.write_bytes(b"not a video")
            with self.assertRaises(ValueError):
                predict_video.run(a, FakeModel)
            self.assertFalse(a.output.exists())


if __name__ == "__main__":
    unittest.main()
