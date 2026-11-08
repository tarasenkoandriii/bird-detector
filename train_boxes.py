#!/usr/bin/env python3
"""Обережне донавчання bbox-гілок; backbone та класифікація зафіксовані."""

import hashlib
import json
from pathlib import Path
from common import ROOT, UkrainianParser, resolve_device


def box_parameter(name, head):
    return name.startswith(f"model.{head}.") and (".cv2." in name or ".one2one_cv2." in name)


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "datasets/bird/data.yaml")
    p.add_argument("--model", type=Path, default=ROOT / "weights/best.pt")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--lr0", type=float, default=0.00001)
    p.add_argument("--name", default="bbox-conservative")
    a = p.parse_args()
    if min(a.epochs, a.imgsz, a.lr0) <= 0 or not a.model.is_file() or not a.data.is_file():
        p.error("Потрібні наявні ваги та додатні параметри.")
    out = ROOT / "runs/experiments" / a.name
    if out.exists():
        p.error("Експеримент уже існує; задайте нове --name.")
    import torch
    from ultralytics import YOLO

    model = YOLO(str(a.model))
    head = len(model.model.model) - 1
    protected = {}

    def setup(t):
        for n, param in t.model.named_parameters():
            trainable = box_parameter(n, head)
            param.requires_grad_(trainable)
            if not trainable:
                protected[n] = param.detach().cpu().clone()
        # Усі BN працюють із збереженою статистикою, bbox convolution навчається.
        for module in t.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        (out / "protocol.json").write_text(
            json.dumps(
                {
                    "source_sha256": hashlib.sha256(a.model.read_bytes()).hexdigest(),
                    "epochs": a.epochs,
                    "imgsz": a.imgsz,
                    "lr0": a.lr0,
                    "note": "Лише bbox-гілки; вибір за validation, test не використовується.",
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    def freeze_bn(t):
        for module in t.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

    def verify(t):
        changed = [
            n
            for n, param in t.model.named_parameters()
            if n in protected and not torch.equal(param.detach().cpu(), protected[n])
        ]
        (out / "protected_parameters.json").write_text(
            json.dumps({"unchanged": not changed, "changed_names": changed})
        )
        if changed:
            raise RuntimeError("Змінилися захищені параметри.")

    model.add_callback("on_train_start", setup)
    model.add_callback("on_train_batch_start", freeze_bn)
    model.add_callback("on_fit_epoch_end", verify)
    model.train(
        data=str(a.data.resolve()),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=8,
        device=resolve_device("cpu"),
        workers=0,
        amp=False,
        freeze=head,
        optimizer="AdamW",
        lr0=a.lr0,
        lrf=0.1,
        warmup_epochs=0,
        warmup_bias_lr=a.lr0,
        weight_decay=0,
        mosaic=0,
        close_mosaic=0,
        scale=0,
        translate=0,
        fliplr=0,
        seed=26,
        deterministic=True,
        plots=False,
        patience=0,
        project=str(out.parent),
        name=a.name,
        save=True,
    )
    print("Кандидат:", model.trainer.best)


if __name__ == "__main__":
    main()
