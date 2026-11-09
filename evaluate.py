#!/usr/bin/env python3
"""Оцінювання найкращих ваг bird на валідації або окремій тестовій вибірці."""

import json
import uuid
from pathlib import Path
from common import UkrainianParser, ROOT, resolve_device
from report_io import dataset_snapshot, sha256, write_json


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "datasets/bird/data.yaml")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/best.pt")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument(
        "--device",
        default="cpu",
        help="CPU — еталонний пристрій для перевірки координат у цьому старому MPS-стеку",
    )
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--report", type=Path, help='Новий JSON; типово у runs/evaluate/')
    p.add_argument("--overwrite", action="store_true", help="Явно дозволити заміну звіту --report")
    a = p.parse_args()
    if a.batch < 1 or a.imgsz < 32 or a.imgsz % 32:
        p.error('--batch має бути додатним, --imgsz — кратним 32 і не меншим за 32.')
    if not a.data.is_file() or not a.weights.is_file():
        p.error("Дані або ваги відсутні. Спочатку виконайте prepare_data.py.")
    out = a.report or ROOT / 'runs/evaluate' / f'{a.split}-{uuid.uuid4().hex}.json'
    if out.resolve() in {a.data.resolve(), a.weights.resolve()}:
        p.error('Звіт не може замінювати датасет або ваги.')
    if out.exists() and not a.overwrite:
        p.error('Звіт уже існує; задайте інший --report або явно додайте --overwrite.')
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from compare_models import bird_validator
    import torch, ultralytics

    yaml_digest = sha256(a.data)
    config = check_det_dataset(str(a.data.resolve()), autodownload=False)
    if config['names'] != {0: 'bird'}:
        p.error('Потрібна однокласова GT names={0: bird}.')
    model = YOLO(str(a.weights))
    source_ids = [int(k) for k, name in model.names.items() if name == 'bird']
    if len(source_ids) != 1:
        p.error('Модель повинна містити рівно один вихідний клас bird.')
    before = dataset_snapshot(config, a.split)
    weights_digest = sha256(a.weights)
    result = model.val(
        validator=bird_validator(source_ids[0]),
        data=str(a.data.resolve()),
        split=a.split,
        device=resolve_device(a.device),
        imgsz=a.imgsz,
        batch=a.batch,
        workers=0,
        conf=0.001,
        iou=0.7,
        max_det=300,
        plots=False,
        project=str(ROOT / "runs/evaluate"),
        name=a.split,
    )
    if (dataset_snapshot(config, a.split) != before or sha256(a.weights) != weights_digest
            or sha256(a.data) != yaml_digest):
        raise RuntimeError('Дані або ваги змінилися під час оцінювання; звіт не збережено.')
    report = {
        "split": a.split,
        "device": a.device,
        "imgsz": a.imgsz,
        "batch": a.batch,
        "weights_sha256": weights_digest,
        "data_yaml_sha256": yaml_digest,
        "dataset": before,
        "evaluator": dict(name='bird-only-v1', source_class=source_ids[0],
                          conf=0.001, iou=0.7, max_det=300,
                          code_sha256=sha256(ROOT / 'compare_models.py')),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "metrics": result.results_dict,
        "speed_ms_per_image": result.speed,
    }
    write_json(out, report, overwrite=a.overwrite)
    print(json.dumps(report, indent=2))
    print('Звіт:', out)


if __name__ == "__main__":
    main()
