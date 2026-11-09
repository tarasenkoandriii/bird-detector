#!/usr/bin/env python3
"""YOLO26s як окремий challenger: той самий head-only рецепт, що у nano.

S не замінює обов'язкову для ТЗ YOLO26n. Контроль rawS → trainedS
відокремлює ефект донавчання від більшої місткості моделі.
"""
from pathlib import Path

from common import ROOT, UkrainianParser


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument('--base', type=Path, default=ROOT / 'yolo26s.pt')
    p.add_argument('--data', type=Path, default=ROOT / 'datasets/bird-video-detection-v2/data.yaml')
    p.add_argument('--name')
    p.add_argument('--epochs', type=int, choices=[3, 5], default=3)
    a = p.parse_args()
    if not a.base.is_file():
        p.error('Потрібен наявний офіційний yolo26s.pt; цей скрипт не завантажує ваги автоматично.')
    if not a.data.is_file():
        p.error('Спочатку підготуйте окрему v2-вибірку.')
    import torch

    checkpoint = torch.load(a.base, map_location='cpu', weights_only=False)
    model = checkpoint.get('ema') or checkpoint.get('model')
    if model is None or model.yaml.get('scale') != 's':
        p.error('Потрібен саме pretrained YOLO26s, не nano або checkpoint іншого масштабу.')
    if not isinstance(model.names, dict) or len(model.names) != 80 or model.names.get(14) != 'bird':
        p.error('Цей контроль починається з вихідної COCO YOLO26s: 80 класів і bird=14.')
    del checkpoint, model

    from train_video_neck import run_recipe

    name = a.name or f'video-v2-small-head-{a.epochs}ep'
    run_recipe(base=a.base, name=name, scope='head', epochs=a.epochs,
               lr0=3e-6, cub_images=32, extra_repeat=3, data=a.data, seed=26)


if __name__ == '__main__':
    main()
