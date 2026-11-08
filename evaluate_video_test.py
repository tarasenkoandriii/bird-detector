#!/usr/bin/env python3
"""Покриття еталонного птаха на фіксованих відеокадрах без зміни моделі."""

import hashlib, json, time
from pathlib import Path
from common import ROOT, UkrainianParser, resolve_device
from inference import SafeDetectionPredictor
from analyze_errors import iou_matrix
import cv2
import numpy as np
from ultralytics import YOLO


def target_metrics(items):
    visible = [x for x in items if x["visible"]]
    values = [x["best_iou"] for x in visible]
    return {
        "sampled_frames": len(items),
        "visible_target_frames": len(visible),
        "excluded_invisible_target_frames": len(items) - len(visible),
        "target_recall_iou50": sum(v >= 0.5 for v in values) / len(values) if values else None,
        "target_recall_iou75": sum(v >= 0.75 for v in values) / len(values) if values else None,
        "mean_best_iou_visible": float(np.mean(values)) if values else None,
        "no_detections_visible": sum(not x["detections"] for x in visible),
    }


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "datasets/video-holdout")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/best.pt")
    p.add_argument("--report", type=Path, default=ROOT / "reports/video_test.json")
    a = p.parse_args()
    original_hash = hashlib.sha256(a.weights.read_bytes()).hexdigest()
    manifest = json.loads((a.data / "manifest.json").read_text())
    model = YOLO(str(a.weights))
    resolve_device("cpu")
    if model.names != {0: "bird"}:
        p.error("Потрібні однокласові ваги bird.")
    # Параметри релізного відеопайплайна, зафіксовані до тесту.
    rows = []
    start = time.perf_counter()
    for i, r in enumerate(manifest["records"]):
        image = cv2.imread(str(a.data / r["image"]))
        if image is None:
            raise ValueError("Не читається кадр.")
        result = model.predict(
            image,
            imgsz=640,
            conf=0.25,
            iou=0.5,
            device="cpu",
            predictor=SafeDetectionPredictor,
            verbose=False,
        )[0]
        pred = result.boxes.data.cpu().numpy()
        if len(pred):
            xywh = pred[:, :4].copy()
            xywh[:, 2:] -= xywh[:, :2]
            keep = cv2.dnn.NMSBoxes(xywh.tolist(), pred[:, 4].tolist(), 0.25, 0.5)
            pred = pred[np.asarray(keep, dtype=int).reshape(-1)]
        x, y, w, h = r["bbox_xywh"]
        gt = [x, y, x + w, y + h]
        best = float(iou_matrix([gt], pred[:, :4]).max()) if len(pred) else 0.0
        rows.append(
            {
                "split": r["split"],
                "sequence": r["sequence"],
                "frame": r["frame"],
                "image": r["image"],
                "visible": r["visible"],
                "gt_xyxy": gt,
                "best_iou": best,
                "detections": pred[:, :5].tolist(),
            }
        )
        if (i + 1) % 50 == 0:
            print("Оцінено кадрів:", i + 1, flush=True)
    report = {
        "weights_sha256": hashlib.sha256(a.weights.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((a.data / "manifest.json").read_bytes()).hexdigest(),
        "source_archive_sha256": manifest["archive_sha256"],
        "source_revision": manifest["revision"],
        "device": "cpu",
        "imgsz": 640,
        "conf": 0.25,
        "post_nms_iou": 0.5,
        "weights_modified": hashlib.sha256(a.weights.read_bytes()).hexdigest() != original_hash,
        "local_finetuning_on_these_videos": False,
        "note": "Оцінюється лише еталонний цільовий птах; інші детекції не оголошуються FP. Precision та mAP не обчислюються. Невидимі цілі виключено з recall/IoU.",
        "splits": {
            s: target_metrics([x for x in rows if x["split"] == s]) for s in ["val", "test"]
        },
        "sequences": {
            s: target_metrics([x for x in rows if x["sequence"] == s])
            for s in sorted({x["sequence"] for x in rows})
        },
        "wall_seconds": time.perf_counter() - start,
        "frames": rows,
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
