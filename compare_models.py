#!/usr/bin/env python3
"""Чесне порівняння COCO YOLO26n та донавченої bird-моделі лише на validation."""

import hashlib
import json
from pathlib import Path
from common import ROOT, UkrainianParser, resolve_device
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import nms


def bird_validator(source_class):
    """Фільтруємо вихідний клас ДО NMS і лише потім перетворюємо його на bird=0."""

    class BirdOnlyValidator(DetectionValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            self.names = {0: "bird"}
            self.nc = 1
            self.metrics.names = self.names

        def postprocess(self, preds):
            outputs = nms.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                classes=[source_class],
                multi_label=True,
                agnostic=False,
                max_det=self.args.max_det,
                end2end=self.end2end,
                max_time_img=1.0,
            )
            return [
                dict(bboxes=x[:, :4], conf=x[:, 4], cls=x[:, 5] * 0, extra=x[:, 6:])
                for x in outputs
            ]

    return BirdOnlyValidator


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "datasets/bird/data.yaml")
    p.add_argument("--base", default="yolo26n.pt")
    p.add_argument("--trained", type=Path, default=ROOT / "weights/best.pt")
    p.add_argument("--sizes", nargs="+", type=int, default=[320, 640])
    p.add_argument("--device", default="cpu")
    p.add_argument("--report", type=Path, default=ROOT / "reports/model_comparison.json")
    p.add_argument(
        "--skip-base",
        action="store_true",
        help="Оцінити лише кандидата; початкова модель уже перевірена",
    )
    a = p.parse_args()
    if not a.data.is_file():
        p.error("Підготуйте дані командою python prepare_data.py.")
    report = {
        "split": "val",
        "device": a.device,
        "batch": 4,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "note": "Лише validation; COCO bird=14 та донавчений bird=0 оцінюються як клас 0. Test не використовується.",
        "runs": [],
    }
    for label, path in (
        [("finetuned", str(a.trained))]
        if a.skip_base
        else [("pretrained", a.base), ("finetuned", str(a.trained))]
    ):
        for size in a.sizes:
            model = YOLO(path)
            bird_ids = [int(k) for k, v in model.names.items() if str(v).casefold() == "bird"]
            if len(bird_ids) != 1:
                raise ValueError("У моделі має бути рівно один вихідний клас bird.")
            source = bird_ids[0]
            metrics = model.val(
                validator=bird_validator(source),
                data=str(a.data.resolve()),
                split="val",
                device=resolve_device(a.device),
                imgsz=size,
                batch=4,
                workers=0,
                conf=0.001,
                iou=0.7,
                max_det=300,
                plots=False,
                project=str(ROOT / "runs/comparison"),
                name=f"{label}-{size}",
            )
            weights = Path(model.ckpt_path)
            report["runs"].append(
                {
                    "model": label,
                    "imgsz": size,
                    "source_class": source,
                    "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                    "metrics": metrics.results_dict,
                    "speed_ms_per_image": metrics.speed,
                }
            )
            a.report.parent.mkdir(parents=True, exist_ok=True)
            a.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            print(
                {"pretrained": "Початкова", "finetuned": "Донавчена"}[label],
                size,
                metrics.results_dict,
                flush=True,
            )
    print("Звіт:", a.report)


if __name__ == "__main__":
    main()
