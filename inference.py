"""CPU coordinate restoration avoids strided in-place MPS operations on torch 2.2."""

import common  # noqa: F401 - configure MPS fallback before importing torch
from ultralytics.models.yolo.detect import DetectionPredictor


class SafeDetectionPredictor(DetectionPredictor):
    """Keep neural inference on the selected device; restore image boxes on CPU.

    torch 2.2.2 MPS produced different xyxy coordinates than CPU on scaled,
    non-square frames. Copy the raw detections before Ultralytics scale_boxes
    performs strided in-place subtraction/division. No package is patched.
    """

    def construct_result(self, pred, img, orig_img, img_path):
        return super().construct_result(pred.detach().cpu().clone(), img, orig_img, img_path)
