#!/usr/bin/env python3
"""Порівняння постобробки NMS та двопрохідної flip-fusion на вибраній вибірці (типово validation)."""

import hashlib, json, time
from pathlib import Path
from common import ROOT, UkrainianParser, resolve_device
from quality import fuse_two_views, unflip_boxes
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator


def quality_validator(mode):
    class QualityValidator(DetectionValidator):
        last_instance = None

        def init_metrics(self, model):
            super().init_metrics(model)
            if dict(model.names) != {0: "bird"}:
                raise ValueError("Потрібна модель одного класу bird.")
            self.quality_model = model
            self.quality_seconds = 0
            type(self).last_instance = self

        def update_metrics(self, preds, batch):
            start = time.perf_counter()
            flipped = (
                super().postprocess(self.quality_model(batch["img"].flip(-1)))
                if mode == "flip"
                else None
            )
            output = []
            for index, primary in enumerate(preds):
                a = torch.cat((primary["bboxes"], primary["conf"][:, None]), 1).cpu().numpy()
                b = np.empty((0, 5))
                if flipped is not None:
                    f = flipped[index]
                    b = torch.cat((f["bboxes"], f["conf"][:, None]), 1).cpu().numpy()
                    b = unflip_boxes(b, batch["img"].shape[-1])
                fused = fuse_two_views(a, b, unmatched_weight=0.5 if mode == "flip" else 1.0)
                rows = torch.tensor(fused, dtype=torch.float32).reshape(-1, 5)
                rows = rows[rows[:, 4] >= self.args.conf]
                output.append(
                    {
                        "bboxes": rows[:, :4],
                        "conf": rows[:, 4],
                        "cls": torch.zeros(len(rows)),
                        "extra": torch.empty((len(rows), 0)),
                    }
                )
            self.quality_seconds += time.perf_counter() - start
            super().update_metrics(output, batch)

    return QualityValidator


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--modes", nargs="+", choices=["nms", "flip"], default=["nms", "flip"])
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--report", type=Path, default=ROOT / "reports/quality_sweep.json")
    a = p.parse_args()
    resolve_device("cpu")
    report = {
        "split": a.split,
        "imgsz": a.imgsz,
        "device": "cpu",
        "weights_sha256": hashlib.sha256((ROOT / "weights/best.pt").read_bytes()).hexdigest(),
        "fusion_match_iou": 0.55,
        "post_nms_iou": 0.5,
        "unmatched_weight": 0.5,
        "note": "Додаткова постобробка й другий forward враховані в extra_quality_ms_per_image; базовий inference-time Ultralytics їх не охоплює.",
        "runs": [],
    }
    for mode in a.modes:
        cls = quality_validator(mode)
        model = YOLO(str(ROOT / "weights/best.pt"))
        start = time.perf_counter()
        metrics = model.val(
            validator=cls,
            data=str(ROOT / "datasets/bird/data.yaml"),
            split=a.split,
            imgsz=a.imgsz,
            batch=4,
            device="cpu",
            workers=0,
            conf=0.001,
            iou=0.7,
            plots=False,
            project=str(ROOT / "runs/quality"),
            name=f"{mode}-{a.imgsz}-{a.split}",
        )
        validator = cls.last_instance
        report["runs"].append(
            {
                "mode": mode,
                "metrics": metrics.results_dict,
                "wall_seconds": time.perf_counter() - start,
                "base_speed_ms_per_image": metrics.speed,
                "extra_quality_ms_per_image": 1000
                * validator.quality_seconds
                / validator.num_images,
            }
        )
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(mode, metrics.results_dict, flush=True)


if __name__ == "__main__":
    main()
