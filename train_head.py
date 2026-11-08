#!/usr/bin/env python3
"""Контрольоване донавчання лише класифікаційної частини без зміни bbox-гілок."""

import copy
import json
from pathlib import Path
from common import ROOT, UkrainianParser, resolve_device


def classifier_parameter(name, head_index):
    prefix = f"model.{head_index}."
    return name.startswith(prefix) and (".cv3." in name or ".one2one_cv3." in name)


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "datasets/bird/data.yaml")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--lr0", type=float, default=0.0001)
    p.add_argument("--device", default="cpu")
    p.add_argument("--name", default="head-only")
    p.add_argument("--model", default="yolo26n.pt")
    p.add_argument("--project", type=Path, default=ROOT / "runs/experiments")
    p.add_argument("--resume", type=Path)
    a = p.parse_args()
    if not a.data.is_file():
        p.error("Спочатку підготуйте дані.")
    if min(a.epochs, a.batch, a.imgsz, a.lr0) <= 0:
        p.error("Параметри мають бути додатними.")
    train_classifier_only(a)


def train_classifier_only(a):
    import torch
    from ultralytics import YOLO

    model = YOLO(str(a.resume) if a.resume else a.model)
    head = len(model.model.model) - 1
    out = a.resume.resolve().parents[1] if a.resume else a.project.resolve() / a.name
    if out.exists() and not a.resume:
        raise ValueError("Експеримент уже існує. Задайте нове --name.")
    protected = {}

    def setup(trainer):
        for name, param in trainer.model.named_parameters():
            param.requires_grad_(classifier_parameter(name, head))
            if not param.requires_grad:
                protected[name] = param.detach().cpu().clone()
        trainer.freeze_layer_names.extend([f"model.{head}.cv2.", f"model.{head}.one2one_cv2."])
        initial = copy.deepcopy(trainer.model).cpu().eval()
        torch.save(
            {"model": initial, "ema": None, "epoch": -1, "train_args": vars(trainer.args)},
            out / "initial-bird.pt",
        )
        (out / "protocol.json").write_text(
            json.dumps(
                {
                    "note": "Лише validation; навчання класифікаційних cv3-гілок, решта параметрів незмінні. Test не використовується.",
                    "head_index": head,
                    "trainable_parameters": sum(
                        p.numel() for p in trainer.model.parameters() if p.requires_grad
                    ),
                    "lr0": a.lr0,
                    "imgsz": a.imgsz,
                    "batch": a.batch,
                    "epochs": a.epochs,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    def verify(trainer):
        changed = [
            n
            for n, p in trainer.model.named_parameters()
            if n in protected and not torch.equal(p.detach().cpu(), protected[n])
        ]
        (out / "protected_parameters.json").write_text(
            json.dumps({"unchanged": not changed, "changed_names": changed}, indent=2)
        )
        if changed:
            raise RuntimeError("Змінилися захищені параметри; експеримент відхилено.")

    model.add_callback("on_train_start", setup)
    model.add_callback("on_fit_epoch_end", verify)
    if a.resume:
        model.train(
            resume=True, data=str(a.data.resolve()), batch=a.batch, device=resolve_device(a.device)
        )
        return Path(model.trainer.best)
    model.train(
        data=str(a.data.resolve()),
        epochs=a.epochs,
        batch=a.batch,
        imgsz=a.imgsz,
        device=resolve_device(a.device),
        workers=0,
        amp=False,
        cache=False,
        freeze=head,
        optimizer="AdamW",
        lr0=a.lr0,
        lrf=0.1,
        warmup_epochs=0,
        warmup_bias_lr=a.lr0,
        weight_decay=0.0001,
        mosaic=0,
        close_mosaic=0,
        scale=0,
        translate=0,
        seed=26,
        deterministic=True,
        val=True,
        plots=False,
        save=True,
        patience=0,
        project=str(a.project.resolve()),
        name=a.name,
    )
    print("Кандидат:", model.trainer.best)
    return Path(model.trainer.best)


if __name__ == "__main__":
    main()
