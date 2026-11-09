#!/usr/bin/env python3
"""Контроль забування фотографій: D0 / head3 / head5, лише CUB validation600."""
import hashlib
import json
import time
from pathlib import Path
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from common import ROOT
from compare_models import bird_validator
from train_video_focus import atomic


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(data):
    cfg = yaml.safe_load(data.read_text())
    folder = Path(cfg['path']) / cfg['val']
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if len(images) != 600:
        raise ValueError(f'Очікували 600 validation images, отримали {len(images)}')
    records = []
    for image in images:
        label = image.parent.parent / 'labels' / f'{image.stem}.txt'
        records.append(dict(image=image.name, image_sha256=sha(image), label_sha256=sha(label)))
    return records


def main():
    data = ROOT / 'datasets/bird-expanded/data.yaml'
    target = ROOT / 'reports/video_study/photo_guard.json'
    before = fingerprint(data)
    report = dict(state='running', split='val', test_used=False, data=str(data),
                  data_yaml_sha256=sha(data), records=before,
                  records_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
                  torch=torch.__version__, ultralytics=ultralytics.__version__,
                  protocol=dict(device='cpu', imgsz=640, batch=4, workers=0,
                                conf=0.001, iou=0.7, max_det=300), runs=[])
    atomic(target, report)
    start = time.time()
    for label, weights in [
        ('D0', ROOT / 'weights/best.pt'),
        ('D-head3', ROOT / 'runs/experiments/video-v2-d-head-3ep/weights/best.pt'),
        ('D-head5', ROOT / 'runs/experiments/video-v2-d-head-5ep/weights/best.pt'),
    ]:
        digest = sha(weights)
        model = YOLO(str(weights))
        if model.names != {0: 'bird'}:
            raise ValueError('Потрібна однокласова bird-модель')
        t = time.time()
        metrics = model.val(validator=bird_validator(0), data=str(data), split='val',
                            plots=False, project=str(ROOT / 'runs/video-photo-guard'),
                            name=label, **report['protocol'])
        if sha(weights) != digest or fingerprint(data) != before:
            raise RuntimeError('Ваги або validation змінилися під час оцінки')
        report['runs'].append(dict(model=label, weights=str(weights), weights_sha256=digest,
                                   metrics=metrics.results_dict, speed_ms_per_image=metrics.speed,
                                   elapsed_seconds=time.time()-t))
        atomic(target, report)
    report.update(state='completed', elapsed_seconds=time.time()-start,
                  validation_unchanged=True, release_weights_unchanged=True)
    atomic(target, report)
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
