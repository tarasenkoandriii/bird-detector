#!/usr/bin/env python3
"""Fine-tune pretrained YOLO26n for one class: bird."""

import argparse
from pathlib import Path
from common import ROOT, resolve_device


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "datasets/bird/data.yaml")
    p.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Pretrained weights, downloaded by Ultralytics if absent",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--device", default="auto", help="auto, cpu, mps, or CUDA device index")
    p.add_argument("--project", type=Path, default=ROOT / "runs/train")
    p.add_argument("--name", default="bird")
    p.add_argument("--resume", type=Path, help="Unfinished last.pt with optimizer state")
    return p


def main():
    p = parser()
    a = p.parse_args()
    if min(a.epochs, a.batch, a.imgsz) < 1 or a.imgsz % 32:
        p.error("Positive epochs/batch and imgsz divisible by 32 are required.")
    if not a.data.is_file():
        p.error(
            f"Dataset YAML missing: {a.data}. Run python prepare_data.py first, or pass --data."
        )
    if a.resume and not a.resume.is_file():
        p.error("Resume checkpoint does not exist.")
    try:
        device = resolve_device(a.device)
        from ultralytics import YOLO

        model = YOLO(str(a.resume) if a.resume else a.model)
        if a.resume:
            model.train(resume=True, data=str(a.data.resolve()), device=device, batch=a.batch)
        else:
            model.train(
                data=str(a.data.resolve()),
                epochs=a.epochs,
                batch=a.batch,
                imgsz=a.imgsz,
                device=device,
                workers=0,
                amp=False,
                cache=False,
                optimizer="AdamW",
                lr0=0.001,
                seed=26,
                deterministic=True,
                mosaic=0,
                close_mosaic=0,
                val=True,
                plots=True,
                save=True,
                save_period=5,
                patience=0,
                project=str(a.project.resolve()),
                name=a.name,
            )
        print(f"Best weights: {model.trainer.best}")
        print("Bundled weights/best.pt was not overwritten.")
    except (ValueError, OSError) as e:
        p.exit(1, f"Error: {e}\n")


if __name__ == "__main__":
    main()
