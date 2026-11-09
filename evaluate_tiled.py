#!/usr/bin/env python3
"""Контроль full-frame проти перекривних фрагментів на validation, без навчання.

Це власний tiled inference за принципом SAHI, не інтеграція пакета sahi.
Протокол: повний кадр + tiles 640 px, overlap=0.20, global NMS IoU=0.70.
Для обох режимів застосовано однаковий global NMS, conf=0.001 та max_det=300.
"""

import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from common import ROOT, UkrainianParser, resolve_device


def axis_windows(length, tile=640, overlap=0.20):
    """Покрити всю вісь без порожніх/дубльованих вікон, вирівнявши останнє."""
    if length <= 0 or tile <= 0 or not 0 <= overlap < 1:
        raise ValueError('Некоректний розмір або overlap.')
    if length <= tile:
        return [(0, length)]
    step = max(1, int(round(tile * (1 - overlap))))
    starts = list(range(0, length - tile + 1, step))
    starts.append(length - tile)
    return [(start, min(start + tile, length)) for start in sorted(set(starts))]


def tile_windows(width, height, tile=640, overlap=0.20):
    return [(x1, y1, x2, y2)
            for y1, y2 in axis_windows(height, tile, overlap)
            for x1, x2 in axis_windows(width, tile, overlap)]


def restore_boxes(boxes, offset, frame_size):
    """Повернути tile xyxy/conf/class у координати кадру, обрізати до меж."""
    result = np.asarray(boxes, dtype=np.float32).reshape(-1, 6).copy()
    if not len(result):
        return result
    x, y = offset
    width, height = frame_size
    result[:, [0, 2]] += x
    result[:, [1, 3]] += y
    result[:, [0, 2]] = result[:, [0, 2]].clip(0, width)
    result[:, [1, 3]] = result[:, [1, 3]].clip(0, height)
    return result[(result[:, 2] > result[:, 0]) & (result[:, 3] > result[:, 1])]


def merge_boxes(boxes, iou=0.70, max_det=300):
    """Однокласовий NMS у повних координатах, результат за спаданням confidence."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 6)
    if not len(boxes):
        return boxes.copy()
    boxes = boxes[np.argsort(-boxes[:, 4], kind='stable')]
    xywh = boxes[:, :4].copy()
    xywh[:, 2:] -= xywh[:, :2]
    selected = np.asarray(cv2.dnn.NMSBoxes(
        xywh.tolist(), boxes[:, 4].tolist(), 0.0, iou)).reshape(-1).astype(int)
    return boxes[selected[:max_det]]


def read_targets(path, width, height):
    result = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        values = [float(x) for x in line.split()]
        if len(values) != 5:
            raise ValueError(f'Потрібна bbox-розмітка YOLO: {path}')
        cls, x, y, w, h = values
        if cls != 0 or not all(math.isfinite(v) for v in values):
            raise ValueError(f'Потрібен один клас bird=0: {path}')
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f'Некоректна нормалізація bbox: {path}')
        result.append([(x-w/2)*width, (y-h/2)*height,
                       (x+w/2)*width, (y+h/2)*height])
    return np.asarray(result, dtype=np.float32).reshape(-1, 4)


def detection_metrics(frames):
    """Ultralytics IoU matching + AP API; аргументи перевірені в локальній версії.

    BaseValidator.match_predictions(pred_cls, true_cls, iou[M,N]); predictions
    у спадному порядку confidence. ap_per_class(tp,conf,pred_cls,target_cls).
    """
    import torch
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import ap_per_class, box_iou

    matcher = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
    tp, conf, pred_cls, target_cls = [], [], [], []
    for frame in frames:
        predictions = np.asarray(frame['predictions'], dtype=np.float32).reshape(-1, 6)
        predictions = predictions[np.argsort(-predictions[:, 4], kind='stable')]
        targets = np.asarray(frame['targets'], dtype=np.float32).reshape(-1, 4)
        correct = np.zeros((len(predictions), 10), dtype=bool)
        if len(predictions) and len(targets):
            overlaps = box_iou(torch.from_numpy(targets), torch.from_numpy(predictions[:, :4]))
            correct = BaseValidator.match_predictions(
                matcher, torch.zeros(len(predictions)), torch.zeros(len(targets)), overlaps).numpy()
        tp.append(correct)
        conf.append(predictions[:, 4])
        pred_cls.append(np.zeros(len(predictions)))
        target_cls.append(np.zeros(len(targets)))
    confidence = np.concatenate(conf) if conf else np.empty(0)
    targets = np.concatenate(target_cls) if target_cls else np.empty(0)
    if not len(targets):
        raise ValueError('Валідація не містить жодної позитивної цілі.')
    if not len(confidence):
        return dict(precision=0.0, recall=0.0, map50=0.0, map50_95=0.0)
    result = ap_per_class(np.concatenate(tp), confidence, np.concatenate(pred_cls), targets,
                          plot=False, names={0: 'bird'})
    return dict(precision=float(result[2].mean()), recall=float(result[3].mean()),
                map50=float(result[5][:, 0].mean()), map50_95=float(result[5].mean()))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    part = path.with_suffix(path.suffix + '.part')
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    part.replace(path)


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'datasets/bird-video-detection-v2/data.yaml')
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT / 'reports/tiled_control')
    a = p.parse_args()
    if not a.data.is_file() or not a.weights.is_file():
        p.error('Потрібні підготовлена v2-валідація та наявні однокласові ваги.')
    if a.out.exists():
        p.error('Каталог результатів уже існує; задайте новий --out.')
    cfg = yaml.safe_load(a.data.read_text())
    names = cfg.get('names')
    if names not in ({0: 'bird'}, ['bird']):
        p.error('Потрібна схема класів {0: bird}.')
    data_root = Path(cfg.get('path') or a.data.resolve().parent)
    if not data_root.is_absolute():
        data_root = (a.data.resolve().parent / data_root).resolve()
    val = Path(cfg['val'])
    image_dir = val if val.is_absolute() else data_root / val
    images = sorted(x for x in image_dir.iterdir() if x.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not images:
        p.error('Validation-зображення відсутні.')
    records = []
    for image in images:
        label = image.parent.parent / 'labels' / f'{image.stem}.txt'
        records.append(dict(image=image.name, image_sha256=sha(image), label_sha256=sha(label)))
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    a.out.mkdir(parents=True)
    protocol = dict(state='prepared', method='manual tiled inference за принципом SAHI',
        split='val', data_yaml=str(a.data.resolve()), data_sha256=fingerprint,
        evaluator_sha256=sha(Path(__file__)),
        weights_sha256=sha(a.weights), classes={0: 'bird'}, device='cpu',
        imgsz=640, conf=0.001, global_nms_iou=0.70, max_det=300,
        tile_size=640, overlap=0.20, tiled_includes_full_frame=True,
        order=['full_frame', 'full_frame_plus_tiles'], records=records,
        note='Фіксований протокол до запуску; без пошуку параметрів. Test не читається. '
             'Це парний контроль predict(batch=1) з однаковим NMS, не підміна старих val(batch=4).')
    write_json(a.out / 'protocol.json', protocol)

    from ultralytics import YOLO
    import ultralytics
    import torch
    from inference import SafeDetectionPredictor

    resolve_device('cpu')
    started = time.perf_counter()
    model = YOLO(str(a.weights))
    if dict(model.names) != {0: 'bird'}:
        raise ValueError('Ваги мають бути однокласовими; COCO bird=14 спочатку remap у bird=0.')

    def predict(image):
        result = model.predict(image, imgsz=640, conf=0.001, iou=0.70,
            max_det=300, classes=[0], device='cpu', rect=True,
            predictor=SafeDetectionPredictor, verbose=False)[0]
        return result.boxes.data.cpu().numpy().copy()

    first = cv2.imread(str(images[0]))
    if first is None:
        raise ValueError(f'Не декодується {images[0]}')
    predict(first)  # один явний прогрів; не включено у прогріті часи режимів
    cold = time.perf_counter() - started
    report = dict(protocol=protocol, ultralytics=ultralytics.__version__,
                  torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__,
                  load_and_first_forward_seconds=cold, modes={})
    for mode in protocol['order']:
        mode_started = time.perf_counter()
        frames = []
        for ip in images:
            image = cv2.imread(str(ip))
            if image is None:
                raise ValueError(f'Не декодується {ip}')
            h, w = image.shape[:2]
            gt = read_targets(ip.parent.parent/'labels'/f'{ip.stem}.txt', w, h)
            t0 = time.perf_counter()
            windows = [(0, 0, w, h)]
            if mode == 'full_frame_plus_tiles':
                # Не дублюємо повний кадр, якщо він сам менший за tile.
                windows += [b for b in tile_windows(w, h) if b != (0, 0, w, h)]
            collected = [restore_boxes(predict(image[y1:y2, x1:x2]), (x1, y1), (w, h))
                         for x1, y1, x2, y2 in windows]
            predictions = merge_boxes(np.concatenate(collected), iou=0.70)
            frames.append(dict(image=ip.name, width=w, height=h, calls=len(windows),
                inference_and_merge_seconds=time.perf_counter()-t0,
                targets=gt.tolist(), predictions=predictions.tolist()))
        elapsed = time.perf_counter()-mode_started
        negatives = [f for f in frames if not len(f['targets'])]
        fp_counts = [sum(b[4] >= 0.25 for b in f['predictions']) for f in negatives]
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mib = rss / (1024*1024 if sys.platform == 'darwin' else 1024)
        report['modes'][mode] = dict(metrics=detection_metrics(frames), images=len(frames),
            wall_seconds=elapsed, images_per_second=len(frames)/elapsed,
            timing_scope='Читання/декодування + інференс + NMS; AP обчислено після цього часу.',
            inference_and_merge_seconds=sum(f['inference_and_merge_seconds'] for f in frames),
            cumulative_process_peak_rss_mib=rss_mib,
            negative_images=len(negatives), negative_images_with_fp_conf025=sum(c>0 for c in fp_counts),
            negative_boxes_conf025=sum(fp_counts), frames=frames)
        write_json(a.out / 'results.json', report)
    report['protocol']['state'] = 'completed'
    report['evaluation_wall_seconds_including_load_warmup_and_metrics'] = time.perf_counter()-started
    write_json(a.out / 'results.json', report)
    write_json(a.out / 'protocol.json', report['protocol'])
    print('Контроль завершено:', a.out, flush=True)


if __name__ == '__main__':
    main()
