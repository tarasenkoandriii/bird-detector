#!/usr/bin/env python3
"""Коротке донавчання всієї голови на перевірених відеокадрах і фото."""

import hashlib, json, random
from pathlib import Path
from common import ROOT, UkrainianParser, resolve_device
import yaml


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "datasets/bird-video/data.yaml")
    p.add_argument("--model", type=Path, default=ROOT / "weights/best.pt")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--name", default="video-head-pilot")
    p.add_argument("--lr0", type=float, default=0.00001)
    p.add_argument("--cub-images", type=int, default=128)
    p.add_argument("--seed", type=int, default=26)
    p.add_argument(
        "--data-seed", type=int, help="Фіксований відбір фото; типово збігається з --seed"
    )
    a = p.parse_args()
    if (
        not a.data.is_file()
        or not a.model.is_file()
        or a.epochs < 1
        or a.lr0 <= 0
        or a.cub_images < 1
    ):
        p.error("Потрібні підготовлені дані, ваги й додатна кількість епох.")
    out = ROOT / "runs/experiments" / a.name
    if out.exists():
        p.error("Запуск уже існує; задайте нове --name.")
    cfg = yaml.safe_load(a.data.read_text())
    root = Path(cfg.get("path") or a.data.resolve().parent)
    if not root.is_absolute():
        root = (a.data.resolve().parent / root).resolve()
    videos = sorted((root / "train/images").glob("*.jpg"))
    photos = sorted((ROOT / "datasets/bird-expanded/train/images").glob("*.jpg"))
    if not videos or not photos:
        p.error("Потрібні відеонабір і розширений CUB.")
    data_seed = a.seed if a.data_seed is None else a.data_seed
    random.Random(data_seed).shuffle(photos)
    photos = photos[: a.cub_images]
    work = ROOT / "work" / a.name
    work.mkdir(parents=True, exist_ok=True)
    (work / "train.txt").write_text("".join(str(x.resolve()) + "\n" for x in videos + photos))
    data = work / "data.yaml"
    data.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": str(work / "train.txt"),
                "val": str(root / "val/images"),
                "names": {0: "bird"},
            }
        )
    )
    import torch
    from ultralytics import YOLO

    model = YOLO(str(a.model))
    head = len(model.model.model) - 1
    protected = {}
    if model.names != {0: "bird"}:
        p.error("Потрібна однокласова bird-модель.")

    def freeze_bn(t):
        for module in t.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

    def setup(t):
        for n, param in t.model.named_parameters():
            param.requires_grad_(n.startswith(f"model.{head}."))
            if not param.requires_grad:
                protected[n] = param.detach().cpu().clone()
        freeze_bn(t)
        (out / "protocol.json").write_text(
            json.dumps(
                {
                    "source_weights_sha256": hashlib.sha256(a.model.read_bytes()).hexdigest(),
                    "video_train_images": len(videos),
                    "cub_train_images": len(photos),
                    "epochs": a.epochs,
                    "imgsz": 640,
                    "batch": 8,
                    "nbs": 8,
                    "lr0": a.lr0,
                    "seed": a.seed,
                    "data_seed": data_seed,
                    "train_selection_sha256": hashlib.sha256(
                        "\n".join(x.name for x in videos + photos).encode()
                    ).hexdigest(),
                    "note": "Навчається вся голова; backbone/neck і BN-статистика зафіксовані. Test не використовується. Релізні ваги не перезаписуються.",
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    def verify(t):
        changed = [
            n
            for n, p in t.model.named_parameters()
            if n in protected and not torch.equal(p.detach().cpu(), protected[n])
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
        data=str(data),
        epochs=a.epochs,
        imgsz=640,
        batch=8,
        nbs=8,
        device=resolve_device("cpu"),
        workers=0,
        freeze=head,
        optimizer="AdamW",
        lr0=a.lr0,
        lrf=0.1,
        warmup_epochs=0,
        warmup_bias_lr=0,
        weight_decay=0,
        mosaic=0,
        close_mosaic=0,
        scale=0,
        translate=0,
        fliplr=0,
        amp=False,
        seed=a.seed,
        deterministic=True,
        patience=0,
        plots=False,
        project=str(out.parent),
        name=a.name,
        save=True,
    )
    print("Кандидат:", model.trainer.best)


if __name__ == "__main__":
    main()
