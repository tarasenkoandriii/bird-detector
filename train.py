#!/usr/bin/env python3
"""Донавчання попередньо навченої YOLO26n для одного класу bird."""

from pathlib import Path
from common import UkrainianParser, ROOT, resolve_device


def parser():
    p = UkrainianParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "datasets/bird/data.yaml")
    p.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Початкові ваги; за відсутності завантажуються Ultralytics",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--device", default="cpu", help="auto, cpu, mps або індекс пристрою CUDA")
    p.add_argument("--project", type=Path, default=ROOT / "runs/train")
    p.add_argument("--name", default="bird")
    p.add_argument("--lr0", type=float, default=0.0001, help="Початкова швидкість навчання")
    p.add_argument(
        "--strategy",
        choices=["head", "full"],
        default="head",
        help="head: лише класифікація; full: усі параметри",
    )
    p.add_argument("--resume", type=Path, help="Незавершений last.pt зі станом оптимізатора")
    return p


def main():
    p = parser()
    a = p.parse_args()
    if min(a.epochs, a.batch, a.imgsz) < 1 or a.imgsz % 32 or a.lr0 <= 0:
        p.error("epochs і batch мають бути додатними, а imgsz — кратним 32.")
    if not a.data.is_file():
        p.error(
            f"YAML датасету відсутній: {a.data}. Спочатку виконайте python prepare_data.py або задайте --data."
        )
    if a.resume and not a.resume.is_file():
        p.error("Файл checkpoint для відновлення не існує.")
    try:
        if a.strategy == "head":
            from train_head import train_classifier_only

            best = train_classifier_only(a)
            print(f"Найкращі ваги: {best}")
            return
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
                lr0=a.lr0,
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
        print(f"Найкращі ваги: {model.trainer.best}")
        print("Додані до проєкту weights/best.pt не перезаписані.")
    except (ValueError, OSError) as e:
        p.exit(1, f"Помилка: {e}\n")


if __name__ == "__main__":
    main()
