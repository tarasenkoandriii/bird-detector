#!/usr/bin/env python3
"""Контрольовані 3 або 5 епох: голова / neck+голова YOLO26n чи YOLO26s на video v2."""

import copy
import hashlib
import json
import os
import random
import re
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from common import ROOT, UkrainianParser, resolve_device
from compare_models import bird_validator
from train_video_focus import atomic, sha


METRIC = 'metrics/mAP50-95(B)'
RECIPE = dict(epochs=3, imgsz=640, batch=8, nbs=8, seed=26,
              optimizer='AdamW', lr0=0.000003, lrf=0.1,
              weight_decay=0, warmup_epochs=0, warmup_bias_lr=0,
              mosaic=0, close_mosaic=0, scale=0, translate=0, fliplr=0,
              hsv_h=0, hsv_s=0, hsv_v=0, amp=False, workers=0,
              deterministic=True, patience=0, plots=False,
              save=True, save_period=1)


def layer_groups(cfg):
    """Межі задає фактичний YAML: neck — між backbone й фінальним Detect."""
    backbone = cfg.get('backbone')
    tail = cfg.get('head')
    if not isinstance(backbone, list) or not backbone or not isinstance(tail, list) or len(tail) < 2:
        raise ValueError('Не вдалося визначити backbone/neck із model YAML.')
    module = tail[-1][2]
    module_name = module if isinstance(module, str) else getattr(module, '__name__', '')
    if module_name.split('.')[-1] != 'Detect':
        raise ValueError('Потрібна стандартна detection-голова Detect.')
    boundary = len(backbone)
    head = boundary + len(tail) - 1
    return dict(backbone=list(range(boundary)), neck=list(range(boundary, head)), head=[head])


def trainable_parameter(name, groups, scope):
    if scope not in {'head', 'neck-head'}:
        raise ValueError('Невідомий scope.')
    match = re.match(r'^model\.(\d+)\.', name)
    if not match or '.dfl.' in name:
        return False
    allowed = groups['head'] + (groups['neck'] if scope == 'neck-head' else [])
    return int(match.group(1)) in allowed


def freeze_bn(model):
    """Фіксує running statistics, зберігаючи вибір affine-параметрів зовнішньому коду."""
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def bn_state(model):
    return {name: value.detach().cpu().clone()
            for name, value in model.named_buffers()
            if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}


def changed_tensors(snapshot, current):
    return [name for name, value in snapshot.items()
            if name not in current or not torch.equal(value, current[name].detach().cpu())]


def class_schema(names):
    if names == {0: 'bird'}:
        return 'single-bird'
    if len(names) == 80 and names.get(14) == 'bird':
        return 'coco80'
    raise ValueError('Потрібні COCO bird=14 або однокласові names={0: bird}.')


def parser():
    p = UkrainianParser(description=__doc__)
    p.add_argument('--base', type=Path, default=ROOT / 'yolo26n.pt')
    p.add_argument('--data', type=Path, default=ROOT / 'datasets/bird-video-detection-v2/data.yaml')
    p.add_argument('--name', default='video-v2-neck-head-3ep')
    p.add_argument('--scope', choices=['head', 'neck-head'], default='neck-head')
    p.add_argument('--seed', type=int, default=26)
    p.add_argument('--data-seed', type=int, default=26)
    p.add_argument('--lr0', type=float, default=0.000003)
    p.add_argument('--cub-images', type=int, default=32)
    p.add_argument('--extra-repeat', type=int, default=3)
    p.add_argument('--epochs', type=int, choices=[3, 5], default=3,
                   help='Перший раунд: 3 епохи; окремий повтор перспективного рецепта: 5.')
    return p


def data_record(image):
    label = image.parent.parent / 'labels' / (image.stem + '.txt')
    return dict(image=str(image), image_sha256=sha(image), label_sha256=sha(label))


def fingerprint(records):
    payload = json.dumps(records, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def select_training_images(images, photos, cub_images=32, extra_repeat=3, data_seed=26):
    """Відбір фото незалежний від seed навчання; extra мають явну кратність."""
    sampled = list(photos)
    random.Random(data_seed).shuffle(sampled)
    extras = [ip for ip in images if ip.name.startswith('extra-')]
    return list(images) + extras * (extra_repeat - 1) + sampled[:cub_images]


def run_recipe(args=None, *, base=None, name=None, scope='neck-head', epochs=3,
               lr0=0.000003, cub_images=32, extra_repeat=3, data=None, seed=26, data_seed=26):
    """Єдиний engine для n/s і head-only / neck+head контрольованих запусків.

    Приймає Namespace від CLI або іменовані аргументи thin-wrapper.
    П'ятиeпоховий повтор починається з того самого --base, без зміни scheduler завершеного 3ep.
    """
    if args is None:
        args = SimpleNamespace(base=Path(base or ROOT / 'yolo26n.pt'),
            data=Path(data or ROOT / 'datasets/bird-video-detection-v2/data.yaml'),
            name=name or f'video-v2-{scope}-{epochs}ep', scope=scope, epochs=epochs,
            seed=seed, data_seed=data_seed, lr0=lr0, cub_images=cub_images, extra_repeat=extra_repeat)
    if args.epochs not in {3, 5} or args.scope not in {'head', 'neck-head'}:
        raise ValueError('Контроль допускає head / neck-head та 3 або 5 епох.')
    seed = getattr(args, 'seed', seed)
    data_seed = getattr(args, 'data_seed', data_seed)
    lr0 = getattr(args, 'lr0', lr0)
    cub_images = getattr(args, 'cub_images', cub_images)
    extra_repeat = getattr(args, 'extra_repeat', extra_repeat)
    if min(seed, data_seed, cub_images) < 0 or extra_repeat < 1 or lr0 <= 0:
        raise ValueError('Некоректні seed / LR / кількість фото / кратність extra.')
    recipe = dict(RECIPE, epochs=args.epochs, seed=seed, lr0=lr0)
    if not args.base.is_file() or not args.data.is_file():
        raise ValueError('Потрібні вихідні ваги й підготовлений data.yaml v2.')
    run = ROOT / 'runs/experiments' / args.name
    work = ROOT / 'work' / args.name
    if run.exists() or work.exists():
        raise ValueError('Запуск уже існує. Задайте новий --name; старі результати не перезаписуються.')
    work.mkdir(parents=True)
    started = time.time()
    state = dict(state='preparing', pid=os.getpid(), name=args.name, scope=args.scope,
                 epochs=args.epochs, completed_epochs=0, started_unix=started)

    def status(**fields):
        state.update(fields, updated_unix=time.time())
        atomic(work / 'status.json', state)

    status()
    try:
        device = resolve_device('cpu')
        cfg = yaml.safe_load(args.data.read_text())
        video = Path(cfg.get('path') or args.data.resolve().parent)
        if not video.is_absolute():
            video = (args.data.resolve().parent / video).resolve()
        images = sorted((video / 'train/images').glob('*.jpg'))
        val = sorted((video / 'val/images').glob('*.jpg'))
        extras = [ip for ip in images if ip.name.startswith('extra-')]
        photos = sorted((ROOT / 'datasets/bird-expanded/train/images').glob('*.jpg'))
        if len(images) != 102 or len(val) != 32 or len(extras) != 15 or len(photos) < cub_images:
            raise ValueError('Контроль потребує 102 video train / 32 val, 15 extra та CUB train.')
        # Типовий контроль: 102 унікальні відеокадри, extra×3, +32 CUB = 164 подачі.
        selected = select_training_images(images, photos, cub_images, extra_repeat, data_seed)
        validation = val + sorted((ROOT / 'datasets/bird-expanded/val/images').glob('*.jpg'))
        if len(validation) != 632:
            raise ValueError('Неочікувана кількість захищених validation-зображень.')
        validation_records = [data_record(ip) for ip in validation]
        protected_sha = {row['image_sha256'] for row in validation_records}
        records = [data_record(ip) for ip in selected]
        if any(row['image_sha256'] in protected_sha for row in records):
            raise ValueError('Точний train/validation leakage.')
        (work / 'train.txt').write_text(''.join(str(ip) + '\n' for ip in selected))
        data = work / 'data.yaml'
        data.write_text(yaml.safe_dump(dict(path=str(video), train=str(work / 'train.txt'),
                                           val=str(video / 'val/images'), names={0: 'bird'})))
        source = YOLO(str(args.base))
        if source.model.yaml.get('scale') not in {'n', 's'}:
            raise ValueError('Контроль допускає лише YOLO26n / YOLO26s.')
        source_schema = class_schema(source.names)
        groups = layer_groups(source.model.yaml)
        if groups != dict(backbone=list(range(11)), neck=list(range(11, 23)), head=[23]):
            raise ValueError('Архітектура відрізняється від перевіреної локальної YOLO26n/s.')
        atomic(work / 'protocol.json', dict(
            recipe=recipe, scope=args.scope, model_scale=source.model.yaml.get('scale'), layer_groups=groups,
            source_weights_sha256=sha(args.base), model_yaml=source.model.yaml,
            dataset_yaml_sha256=sha(args.data), train_exposures=len(selected),
            unique_video_images=102, extra_repeat=extra_repeat, cub_images=cub_images, data_seed=data_seed, source_schema=source_schema,
            training_records=records, validation_records=validation_records,
            training_sha256=fingerprint(records), validation_sha256=fingerprint(validation_records),
            note='Лише validation v2. Test не використовується. Релізні ваги не змінюються.'))
        # Не порівнюємо v2 з legacy GT. Обидві моделі оцінюємо на тих самих v2 labels.
        val_args = dict(data=str(data), split='val', device=device, imgsz=640, batch=4,
                        workers=0, conf=0.001, iou=0.7, max_det=300, plots=False,
                        project=str(work / 'checks'))
        initial = work / 'initial-bird.pt'
        if source_schema == 'coco80':
            base_result = source.val(validator=bird_validator(14), name='base-coco-v2', **val_args)
            source = YOLO(str(args.base))  # val може fuse-ити модель; remap із чистого checkpoint
            one = DetectionModel(copy.deepcopy(source.model.yaml), ch=3, nc=1, verbose=False)
            one.names = {0: 'bird'}
            one.load(source.model)
            source.model = one
            source.save(str(initial))
            initial_result = YOLO(str(initial)).val(name='initial-bird-v2', **val_args)
            baseline = float(initial_result.results_dict[METRIC])
            base_score = float(base_result.results_dict[METRIC])
            if abs(baseline - base_score) > 1e-5:
                raise RuntimeError(f'Remap змінив початкову v2 mAP: {baseline} != {base_score}')
        else:
            # Для D не перебудовуємо голову й не переносимо клас повторно.
            shutil.copyfile(args.base, initial)
            base_result = YOLO(str(initial)).val(name='base-bird-v2', **val_args)
            initial_result = base_result
            baseline = float(base_result.results_dict[METRIC])
            if sha(initial) != sha(args.base):
                raise RuntimeError('Однокласові стартові ваги змінено під час копіювання.')
        atomic(work / 'baseline.json', dict(base=base_result.results_dict,
               initial_bird=initial_result.results_dict, source_schema=source_schema,
               source_weights_sha256=sha(args.base), initial_weights_sha256=sha(initial),
               validation_sha256=fingerprint(validation_records)))
        model = YOLO(str(initial))
        initial_state = {name: value.detach().cpu().clone()
                         for name, value in model.model.state_dict().items()}
        fixed = {}
        buffers = {}

        def freeze_callback(trainer):
            freeze_bn(trainer.model)

        def setup(trainer):
            for name, param in trainer.model.named_parameters():
                param.requires_grad_(trainable_parameter(name, groups, args.scope))
                if not param.requires_grad:
                    fixed[name] = param.detach().cpu().clone()
            buffers.update(bn_state(trainer.model))
            freeze_bn(trainer.model)
            zero = {}
            for label, network in [('live', trainer.model), ('ema', trainer.ema.ema)]:
                changed = changed_tensors(initial_state, network.state_dict())
                zero[label] = dict(unchanged=not changed, differences=changed)
                if changed:
                    raise RuntimeError(f'Зміна {label} до optimizer step: {changed}')
            atomic(work / 'zero_step.json', zero)
            status(state='training', baseline_video_map50_95=baseline)

        def verify(trainer):
            changed = changed_tensors(fixed, dict(trainer.model.named_parameters()))
            changed += changed_tensors(buffers, dict(trainer.model.named_buffers()))
            if changed:
                raise RuntimeError('Змінилися захищені параметри/BN: ' + str(changed))
            atomic(work / 'protected.json', dict(unchanged=True, bn_buffers_unchanged=True))
            status(completed_epochs=min(args.epochs, int(trainer.epoch) + 1),
                   video_metrics={key: float(value) for key, value in trainer.metrics.items()})

        model.add_callback('on_train_start', setup)
        model.add_callback('on_train_batch_start', freeze_callback)
        model.add_callback('on_fit_epoch_end', verify)
        first_active = min(groups['head'] if args.scope == 'head' else groups['neck'])
        result = model.train(data=str(data), device=device, freeze=first_active,
                             project=str(run.parent), name=args.name, **recipe)
        if fingerprint([data_record(ip) for ip in selected]) != fingerprint(records):
            raise RuntimeError('Train-дані або labels змінилися протягом запуску.')
        if fingerprint([data_record(ip) for ip in validation]) != fingerprint(validation_records):
            raise RuntimeError('Validation-дані або labels змінилися протягом запуску.')
        metrics = {key: float(value) for key, value in result.results_dict.items()}
        trained_score = metrics[METRIC]
        # Нульова епоха теж кандидат: «best після навчання» не обов'язково кращий за старт.
        improved = trained_score > baseline + 1e-8
        status(state='trained', best_trained=str(model.trainer.best),
               selected=str(model.trainer.best) if improved else str(initial),
               initial_retained=not improved, best_video_metrics=metrics,
               video_gain_pp=(trained_score - baseline) * 100,
               release_replaced=False, elapsed_seconds=time.time() - started)
        return dict(state)
    except BaseException as exc:
        status(state='failed', error=str(exc))
        raise


def main():
    args = parser().parse_args()
    print(json.dumps(run_recipe(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
