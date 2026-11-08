"""Regression: old MPS strided operations must not corrupt restored image boxes."""

import unittest
from types import SimpleNamespace
import numpy as np
import torch
from inference import SafeDetectionPredictor


class Coordinates(unittest.TestCase):
    def test_restore_non_square_image_on_cpu_and_available_mps(self):
        predictor = object.__new__(SafeDetectionPredictor)
        predictor.model = SimpleNamespace(names={0: "bird"})
        original = np.zeros((360, 640, 3), dtype=np.uint8)
        # Letterbox image is 192x320: scale .5 and vertical padding 6.
        expected = torch.tensor([[200.0, 68.0, 480.0, 308.0, 0.9, 0.0]])
        devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                raw = torch.tensor([[100.0, 40.0, 240.0, 160.0, 0.9, 0.0]], device=device)
                result = predictor.construct_result(
                    raw, torch.empty(1, 3, 192, 320), original, "test.jpg"
                )
                self.assertEqual(result.boxes.data.device.type, "cpu")
                self.assertTrue(torch.allclose(result.boxes.data, expected, atol=0.001))


if __name__ == "__main__":
    unittest.main()
