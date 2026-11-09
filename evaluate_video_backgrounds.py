#!/usr/bin/env python3
"""Хибні детекції на перевірених негативних зображеннях відеовалідації."""

import hashlib, json, uuid
from pathlib import Path
from common import ROOT, UkrainianParser, resolve_device
from inference import SafeDetectionPredictor
from ultralytics import YOLO
import cv2, numpy as np
from report_io import write_json, sha256


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "datasets/bird-video")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/best.pt")
    p.add_argument("--report", type=Path, help='Новий JSON-звіт; типово runs/backgrounds-*.json')
    p.add_argument('--overwrite', action='store_true', help='Явно дозволити заміну звіту')
    a = p.parse_args()
    target = a.report or ROOT / 'runs' / f'backgrounds-{uuid.uuid4().hex}.json'
    if target.resolve() == a.weights.resolve():
        p.error('Звіт не може замінювати ваги.')
    if target.exists() and not a.overwrite:
        p.error('Звіт уже існує; задайте інший --report або --overwrite.')
    metadata = json.loads((a.data / "source.json").read_text())
    rows = [r for r in metadata["records"] if r["split"] == "val" and r["kind"] != "positive"]
    if not rows:
        p.error("Немає перевірених негативних кадрів.")
    model = YOLO(str(a.weights))
    if model.names != {0: 'bird'}:
        p.error('Потрібні однокласові ваги bird=0.')
    resolve_device("cpu")
    out = []
    records = []
    for r in rows:
        ip = a.data / r["output_image"]
        label = ip.parent.parent / "labels" / f"{ip.stem}.txt"
        if label.read_text().strip():
            raise ValueError("Негативний приклад має непорожню розмітку.")
        records.append(dict(image=r['output_image'], image_sha256=sha256(ip), label_sha256=sha256(label)))
        result = model.predict(
            str(ip),
            imgsz=640,
            conf=0.25,
            iou=0.5,
            device="cpu",
            predictor=SafeDetectionPredictor,
            verbose=False,
        )[0]
        boxes = result.boxes.data.cpu().numpy()
        count = 0
        if len(boxes):
            xywh = boxes[:, :4].copy()
            xywh[:, 2:] -= xywh[:, :2]
            count = len(
                np.asarray(
                    cv2.dnn.NMSBoxes(xywh.tolist(), boxes[:, 4].tolist(), 0.25, 0.5)
                ).reshape(-1)
            )
        out.append({"image": r["output_image"], "false_positive_boxes": count})
    report = {
        "weights_sha256": hashlib.sha256(a.weights.read_bytes()).hexdigest(),
        "split": "val",
        "images": len(out),
        "images_with_false_positives": sum(r["false_positive_boxes"] > 0 for r in out),
        "false_positive_boxes": sum(r["false_positive_boxes"] for r in out),
        "conf": 0.25,
        "imgsz": 640,
        "post_nms_iou": 0.5,
        "dataset": dict(sha256=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
                        images=len(records), records=records),
        "evaluator": dict(name='bird-background-v1', conf=0.25, iou=0.5, imgsz=640,
                          code_sha256=sha256(Path(__file__))),
        "note": "Негативна валідація містить перевірені фонові вирізки, не повні негативні ролики.",
        "frames": out,
    }
    write_json(target, report, overwrite=a.overwrite)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
