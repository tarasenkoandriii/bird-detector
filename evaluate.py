#!/usr/bin/env python3
"""Evaluate the bundled bird weights on validation or held-out test."""

import argparse, hashlib, json
from pathlib import Path
from common import ROOT, resolve_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "datasets/bird/data.yaml")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/best.pt")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument(
        "--device", default="cpu", help="CPU is the reference for coordinates on this old MPS stack"
    )
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    if not a.data.is_file() or not a.weights.is_file():
        p.error("Data or weights missing. Run prepare_data.py first.")
    from ultralytics import YOLO
    import torch, ultralytics

    result = YOLO(str(a.weights)).val(
        data=str(a.data.resolve()),
        split=a.split,
        device=resolve_device(a.device),
        imgsz=a.imgsz,
        batch=a.batch,
        workers=0,
        plots=False,
        project=str(ROOT / "runs/evaluate"),
        name=a.split,
    )
    report = {
        "split": a.split,
        "device": a.device,
        "imgsz": a.imgsz,
        "batch": a.batch,
        "weights_sha256": hashlib.sha256(a.weights.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "metrics": result.results_dict,
        "speed_ms_per_image": result.speed,
    }
    out = a.report or ROOT / "reports" / f"{a.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
