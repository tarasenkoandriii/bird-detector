#!/usr/bin/env python3
"""Аналіз пропусків і локалізації на validation за фіксованих conf та IoU."""

import json
from pathlib import Path
import cv2
import numpy as np
from common import ROOT, UkrainianParser, resolve_device
from inference import SafeDetectionPredictor
from ultralytics import YOLO


def iou_matrix(a, b):
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(hi - lo, 0).prod(2)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def match(a, b, threshold=0.5):
    iou = iou_matrix(a, b)
    candidates = sorted(
        [(iou[i, j], i, j) for i, j in zip(*np.where(iou >= threshold))], reverse=True
    )
    used_a = set()
    used_b = set()
    values = []
    for score, i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            values.append(float(score))
    return len(values), len(b) - len(values), len(a) - len(values), values


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument(
        "--skip-base", action="store_true", help="Не повторювати незмінну початкову модель"
    )
    a = p.parse_args()
    images = sorted((ROOT / "datasets/bird/val/images").iterdir())
    report = {
        "split": "val",
        "conf": a.conf,
        "iou_match": 0.5,
        "imgsz": a.imgsz,
        "images": len(images),
        "note": "Порівняння з наданими bbox, не повна розмітка всіх можливих птахів у сцені; test не використовується.",
        "models": {},
    }
    resolve_device("cpu")
    for label, path in (
        [("finetuned", str(ROOT / "weights/best.pt"))]
        if a.skip_base
        else [("pretrained", "yolo26n.pt"), ("finetuned", str(ROOT / "weights/best.pt"))]
    ):
        model = YOLO(path)
        source = [k for k, v in model.names.items() if v == "bird"][0]
        rows = []
        tp = fp = fn = empty = localization = 0
        matched = []
        for index, p in enumerate(images):
            image = cv2.imread(str(p))
            h, w = image.shape[:2]
            gt = []
            for line in (p.parents[1] / "labels" / f"{p.stem}.txt").read_text().splitlines():
                _, x, y, bw, bh = map(float, line.split())
                gt.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
            result = model.predict(
                image,
                device="cpu",
                imgsz=a.imgsz,
                conf=a.conf,
                classes=[source],
                predictor=SafeDetectionPredictor,
                verbose=False,
            )[0]
            pred = result.boxes.xyxy.cpu().numpy()
            t, f, n, ious = match(gt, pred)
            tp += t
            fp += f
            fn += n
            empty += not len(pred)
            matched.extend(ious)
            best_iou = float(iou_matrix(gt, pred).max()) if gt and len(pred) else 0
            localization += bool(len(pred) and 0.1 <= best_iou < 0.5)
            rows.append(
                {
                    "image": p.name,
                    "tp": t,
                    "fp": f,
                    "fn": n,
                    "predictions": len(pred),
                    "best_iou": best_iou,
                }
            )
            if (index + 1) % 50 == 0:
                print(
                    {"pretrained": "Початкова", "finetuned": "Донавчена"}[label],
                    index + 1,
                    flush=True,
                )
        report["models"][label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "images_without_detections": empty,
            "localization_failures": localization,
            "mean_matched_iou": float(np.mean(matched)) if matched else 0,
            "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0,
            "per_image": rows,
        }
        (ROOT / "reports/error_analysis.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
    print("Звіт: reports/error_analysis.json")


if __name__ == "__main__":
    main()
