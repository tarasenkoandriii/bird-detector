"""Відновлення координат на CPU обходить некоректні операції MPS у torch 2.2."""

import common  # noqa: F401 - налаштувати MPS fallback до імпорту torch
from ultralytics.models.yolo.detect import DetectionPredictor


class SafeDetectionPredictor(DetectionPredictor):
    """Нейромережа працює на вибраному пристрої, координати відновлюються на CPU.

    На прямокутних кадрах torch 2.2.2 MPS давав інші xyxy, ніж CPU.
    Копіюємо детекції до віднімання та ділення в Ultralytics scale_boxes.
    Встановлені пакети не змінюються.
    """

    def construct_result(self, pred, img, orig_img, img_path):
        return super().construct_result(pred.detach().cpu().clone(), img, orig_img, img_path)
