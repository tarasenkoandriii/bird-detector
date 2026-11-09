#!/usr/bin/env python3
"""Початкова YOLO26n на тих самих фото- й відеовалідаціях, без навчання."""
import hashlib
import json
from pathlib import Path

import yaml
import torch
import ultralytics
from common import ROOT, resolve_device
from compare_models import bird_validator
from ultralytics import YOLO


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = ROOT / 'reports/pretrained_final_validation.json'
    if target.exists():
        raise SystemExit('Оцінку вже збережено; повторний запуск не потрібен.')
    weights = ROOT / 'yolo26n.pt'
    model = YOLO(str(weights))
    bird = [int(k) for k, v in model.names.items() if v == 'bird']
    if bird != [14]:
        raise ValueError('Потрібні вихідні COCO-ваги з bird=14.')
    resolve_device('cpu')
    report = dict(weights_sha256=sha(weights), source_class=14, target_class=0,
                  torch=torch.__version__, ultralytics=ultralytics.__version__,
                  split='val', device='cpu', imgsz=640, batch=4,
                  conf=0.001, iou=0.7, max_det=300, datasets={})
    for name, relative in [('cub', 'datasets/bird-expanded/data.yaml'),
                           ('video', 'datasets/bird-video/data.yaml')]:
        config = ROOT / relative
        data = yaml.safe_load(config.read_text())
        folder = Path(data['path']) / data['val']
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        records = []
        for ip in images:
            label = ip.parent.parent / 'labels' / (ip.stem + '.txt')
            records.append(dict(image=ip.name, image_sha256=sha(ip), label_sha256=sha(label)))
        if len(images) != {'cub': 600, 'video': 32}[name]:
            raise ValueError('Неочікувана кількість validation-зображень.')
        result = YOLO(str(weights)).val(
            validator=bird_validator(14), data=str(config), split='val',
            device='cpu', imgsz=640, batch=4, workers=0, conf=0.001,
            iou=0.7, max_det=300, plots=False,
            project=str(ROOT / 'runs/pretrained-final-validation'), name=name)
        report['datasets'][name] = dict(images=len(images), records=records,
                                       metrics=result.results_dict)
    temp = target.with_suffix('.json.part')
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    temp.replace(target)
    print('Збережено:', target, flush=True)


if __name__ == '__main__':
    main()
