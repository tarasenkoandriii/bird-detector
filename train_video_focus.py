#!/usr/bin/env python3
"""Відеоадаптація bbox-гілок від початкової YOLO26n, без заміни релізних ваг."""
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

import yaml
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from common import ROOT, UkrainianParser, resolve_device
from train_boxes import box_parameter


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, obj):
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    temp.replace(path)


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument('--base', type=Path, default=ROOT / 'yolo26n.pt')
    p.add_argument('--name', default='video-focus-pretrained-bbox')
    p.add_argument('--data', type=Path, default=ROOT / 'datasets/bird-video/data.yaml')
    p.add_argument('--branch', choices=['bbox','head'], default='bbox')
    p.add_argument('--extra-repeat', type=int, default=1)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--lr0', type=float, default=0.00001)
    p.add_argument('--cub-images', type=int, default=64)
    p.add_argument('--seed', type=int, default=26)
    a = p.parse_args()
    if a.epochs < 1 or a.lr0 <= 0 or a.cub_images < 0 or a.extra_repeat < 1 or not a.base.is_file() or not a.data.is_file():
        p.error('Потрібні початкові ваги та коректні параметри.')
    run = ROOT / 'runs/experiments' / a.name
    work = ROOT / 'work' / a.name
    if run.exists() or work.exists():
        p.error('Запуск уже існує; задайте новий --name.')
    work.mkdir(parents=True)
    status_path = work / 'status.json'
    started = time.time()
    state = dict(state='preparing', pid=os.getpid(), name=a.name,
                 completed_epochs=0, epochs=a.epochs, started_unix=started)

    def status(**fields):
        state.update(fields, updated_unix=time.time())
        atomic(status_path, state)

    status()
    try:
        device = resolve_device('cpu')
        cfg = yaml.safe_load(a.data.read_text())
        video = Path(cfg.get('path') or a.data.resolve().parent)
        if not video.is_absolute():
            video = (a.data.resolve().parent / video).resolve()
        images = sorted((video / 'train/images').glob('*.jpg'))
        photos = sorted((ROOT / 'datasets/bird-expanded/train/images').glob('*.jpg'))
        if len(images) < 87 or len(photos) < a.cub_images:
            raise ValueError('Потрібні підготовлені 87 відеоприкладів і CUB train.')
        random.Random(a.seed).shuffle(photos)
        extras = [ip for ip in images if ip.name.startswith('extra-')]
        selected = images + extras * (a.extra_repeat - 1) + photos[:a.cub_images]
        # Перевіряємо точні перетини за зображеннями з двома validation.
        validation = sorted((video / 'val/images').glob('*.jpg'))
        validation += sorted((ROOT / 'datasets/bird-expanded/val/images').glob('*.jpg'))
        if len(validation) != 632:
            raise ValueError('Неочікувані validation-вибірки.')
        protected_sha = {sha(ip) for ip in validation}
        records = []
        for ip in selected:
            label = ip.parent.parent / 'labels' / (ip.stem + '.txt')
            digest = sha(ip)
            if digest in protected_sha:
                raise ValueError('Знайдено точний train/validation leakage.')
            records.append(dict(image=str(ip), image_sha256=digest, label_sha256=sha(label)))
        (work / 'train.txt').write_text(''.join(str(ip) + '\n' for ip in selected))
        data = work / 'data.yaml'
        data.write_text(yaml.safe_dump(dict(path=str(video), train=str(work / 'train.txt'),
                                           val=str(video / 'val/images'), names={0:'bird'})))
        source = YOLO(str(a.base))
        if source.names.get(14) != 'bird' or len(source.names) != 80:
            raise ValueError('Потрібна початкова COCO YOLO26n із bird=14.')
        one = DetectionModel(copy.deepcopy(source.model.yaml), ch=3, nc=1, verbose=False)
        one.names = {0:'bird'}
        one.load(source.model)
        source.model = one
        initial = work / 'initial-bird.pt'
        source.save(str(initial))
        # Перевірка класового перенесення на тій самій відеовалідації ДО навчання.
        baseline = YOLO(str(initial)).val(data=str(data), device=device, imgsz=640,
            batch=4, workers=0, conf=0.001, iou=0.7, max_det=300, plots=False,
            project=str(work / 'checks'), name='initial')
        measured = baseline.results_dict['metrics/mAP50-95(B)']
        reference = json.loads((ROOT / 'reports/pretrained_final_validation.json').read_text())
        expected = reference['datasets']['video']['metrics']['metrics/mAP50-95(B)']
        if abs(measured - expected) > 1e-5:
            raise ValueError(f'Перенесення bird змінило початкову mAP: {measured} != {expected}')
        protocol = dict(source_weights_sha256=sha(a.base), initial_weights_sha256=sha(initial),
            video_train_images=len(images), cub_train_images=a.cub_images,
            epochs=a.epochs, lr0=a.lr0, seed=a.seed, imgsz=640, batch=8, nbs=8,
            optimizer='AdamW', device=device, branch=a.branch, extra_repeat=a.extra_repeat, unique_video_images=len(images), train_exposures=len(selected),
            bn_statistics_frozen=True, records=records, baseline_video_metrics=baseline.results_dict,
            note='Нове донавчання від вихідної моделі. Test не використовується; реліз не замінюється.')
        atomic(work / 'protocol.json', protocol)
        model = YOLO(str(initial))
        initial_state = {n:v.detach().cpu().clone() for n,v in model.model.state_dict().items()}
        head = len(model.model.model) - 1
        fixed, buffers = {}, {}

        def freeze_bn(t):
            for m in t.model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()

        def setup(t):
            for name, param in t.model.named_parameters():
                param.requires_grad_(box_parameter(name, head) if a.branch == 'bbox' else name.startswith(f'model.{head}.'))
                if not param.requires_grad:
                    fixed[name] = param.detach().cpu().clone()
            for name, value in t.model.named_buffers():
                if name.endswith(('running_mean','running_var','num_batches_tracked')):
                    buffers[name] = value.detach().cpu().clone()
            freeze_bn(t)
            zero_step = {}
            for label, network in [('live',t.model),('ema',t.ema.ema)]:
                current_state = network.state_dict()
                differences = [n for n,v in initial_state.items() if n not in current_state or not torch.equal(v,current_state[n].detach().cpu())]
                zero_step[label] = dict(unchanged=not differences, differences=differences)
                if differences:
                    raise RuntimeError('Зміна ваг до першого кроку: '+label+str(differences))
            atomic(work / 'zero_step.json', zero_step)
            status(state='training')

        def verify(t):
            params = dict(t.model.named_parameters())
            current = dict(t.model.named_buffers())
            changed = [n for n,v in fixed.items() if not torch.equal(v, params[n].detach().cpu())]
            changed += [n for n,v in buffers.items() if not torch.equal(v, current[n].detach().cpu())]
            if changed:
                raise RuntimeError('Змінилися захищені параметри/BN: ' + str(changed))
            status(completed_epochs=min(a.epochs, int(t.epoch)+1),
                   video_metrics={k:float(v) for k,v in t.metrics.items()})
            atomic(work / 'protected.json', dict(unchanged=True, bn_buffers_unchanged=True))

        model.add_callback('on_train_start', setup)
        model.add_callback('on_train_batch_start', freeze_bn)
        model.add_callback('on_fit_epoch_end', verify)
        model.train(data=str(data), epochs=a.epochs, imgsz=640, batch=8, nbs=8,
            device=device, workers=0, freeze=head, optimizer='AdamW', lr0=a.lr0,
            lrf=0.1, warmup_epochs=0, warmup_bias_lr=0, weight_decay=0,
            mosaic=0, close_mosaic=0, scale=0, translate=0, fliplr=0,
            hsv_h=0, hsv_s=0, hsv_v=0, amp=False, seed=a.seed, deterministic=True,
            patience=0, plots=False, project=str(run.parent), name=a.name,
            save=True, save_period=1)
        status(state='trained', best=str(model.trainer.best), elapsed_seconds=time.time()-started)
    except BaseException as e:
        status(state='failed', error=str(e))
        raise


if __name__ == '__main__':
    main()
