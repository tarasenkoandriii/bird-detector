#!/usr/bin/env python3
"""Окремий full-visible-bird v2: 102 train / 32 development-validation.

Використовує лише явну ручну розмітку асистента, без прогнозів моделей.
Legacy та вихідні зображення не змінюються.
"""
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image
import yaml

from common import ROOT, UkrainianParser
from tools.dataset import audit


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT / 'datasets/bird-video-focus')
    p.add_argument('--annotations', type=Path,
                   default=ROOT / 'tools/video_full_validation_annotations.json')
    p.add_argument('--out', type=Path, default=ROOT / 'datasets/bird-video-detection-v2')
    a = p.parse_args()
    if a.out.exists():
        p.error('Результат уже існує; legacy та v2 не перезаписуються.')
    annotations = json.loads(a.annotations.read_text())
    if annotations.get('author') != 'assistant' or annotations.get('predictions_used') is not False:
        p.error('Потрібен явний provenance ручної розмітки без прогнозів.')
    rows = annotations['records']
    if Counter(r['split'] for r in rows) != Counter(train=102, val=32):
        p.error('Потрібні 102 train і 32 validation.')
    seqs = {s: {r['sequence'] for r in rows if r['split'] == s} for s in ['train', 'val']}
    if seqs['train'] & seqs['val'] or seqs['val'] != {'bird-7', 'bird-8'}:
        p.error('Порушено розділення відеопослідовностей.')
    if seqs['train'] != {'bird-1', 'bird-6', 'bird-18', 'bird-4', 'bird-12', 'bird-13', 'bird-20'}:
        p.error('Неочікувані навчальні відеопослідовності.')
    work = ROOT / 'work/annotation-v2'
    work.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='prepare-', dir=work))
    try:
        seen = set()
        checksums = {'train': set(), 'val': set()}
        manifest_rows = []
        negatives_unchanged = 0
        revised = Counter()
        for row in rows:
            relative = Path(row['image'])
            if relative.is_absolute() or '..' in relative.parts or relative.parts[:2] != (row['split'], 'images'):
                raise ValueError('Некоректний відносний шлях зображення.')
            if str(relative) in seen:
                raise ValueError('Повторний запис зображення.')
            seen.add(str(relative))
            source = a.source / relative
            source_label = source.parent.parent / 'labels' / (source.stem + '.txt')
            if sha(source) != row['image_sha256'] or sha(source_label) != row['old_label_sha256']:
                raise ValueError('Змінилося вихідне зображення або розмітка: ' + str(relative))
            with Image.open(source) as image:
                width, height = image.size
                if [width, height] != row['size']:
                    raise ValueError('Змінився розмір кадру.')
            target = stage / relative
            label = target.parent.parent / 'labels' / (target.stem + '.txt')
            target.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            lines = []
            for box in row['new_boxes_xywh']:
                if len(box) != 4 or not all(math.isfinite(v) for v in box):
                    raise ValueError('Некоректний bbox.')
                x, y, bw, bh = box
                if not (0 <= x < x+bw <= width and 0 <= y < y+bh <= height):
                    raise ValueError('BBox за межами кадру: ' + str(relative))
                lines.append(f'0 {(x+bw/2)/width:.12f} {(y+bh/2)/height:.12f} {bw/width:.12f} {bh/height:.12f}\n')
            changed = len(row['old_boxes_xywh']) != len(row['new_boxes_xywh']) or any(
                abs(x-y) > 1e-6
                for old, new in zip(row['old_boxes_xywh'], row['new_boxes_xywh'])
                for x, y in zip(old, new)
            )
            if changed:
                label.write_text(''.join(lines))
                revised[row['split']] += 1
            else:
                shutil.copyfile(source_label, label)
            if not lines:
                if source_label.read_text().strip() or label.read_bytes() != source_label.read_bytes():
                    raise ValueError('Негативна розмітка змінилася.')
                if row['split'] == 'val':
                    negatives_unchanged += 1
            if sha(target) != row['image_sha256']:
                raise ValueError('Байти зображення змінилися під час копіювання.')
            checksums[row['split']].add(sha(target))
            manifest_rows.append({**row, 'label': str(label.relative_to(stage)),
                                  'label_sha256': sha(label), 'annotation_changed': changed})
        if checksums['train'] & checksums['val'] or negatives_unchanged != 8:
            raise ValueError('Перетин split або неправильна кількість негативів.')
        (stage / 'data.yaml').write_text(yaml.safe_dump(
            dict(path=str(a.out.resolve()), train='train/images', val='val/images',
                 names={0: 'bird'}, nc=1), sort_keys=False))
        audit_report = audit(stage, distance=4)
        if audit_report['errors'] or any(pair['leakage'] for pair in audit_report['pairs']):
            raise ValueError('Аудит YOLO/міжвибіркових збігів не пройшов.')
        # Аудитні шляхи мають указувати на фінальний dataset, а не тимчасовий stage.
        report_text = json.dumps(audit_report, ensure_ascii=False, indent=2)
        (stage / 'audit.json').write_text(report_text.replace(str(stage.resolve()), str(a.out.resolve())))
        manifest = dict(version=annotations['dataset_version'], author='assistant',
                        human_audit=False, predictions_used=False,
                        annotation_scope='full visible bird; all birds; exclude logos/reflections/prey',
                        policy_uk=annotations['policy_uk'],
                        validation_role='development validation; not independent test',
                        annotation_file=str(a.annotations.relative_to(ROOT)),
                        annotation_sha256=sha(a.annotations),
                        counts={'train': 102, 'val': 32},
                        bbox_counts=dict(Counter(r['split'] for r in manifest_rows for _ in r['new_boxes_xywh'])),
                        revised_annotation_counts=dict(revised), validation_negatives_unchanged=8,
                        image_bytes_unchanged=True, source_sequences={s: sorted(v) for s,v in seqs.items()},
                        cross_split_exact_duplicates=0,
                        cross_split_near_candidates=sum(p['leakage'] for p in audit_report['pairs']),
                        limitations=annotations['limitations'], records=manifest_rows)
        (stage / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        (stage / 'README.md').write_text(
            '# Bird video detection v2\n\n'
            '102 train / 32 development-validation. Розмітка асистента за оригінальними кадрами, '
            'без передбачень моделей; це не незалежний людський аудит.\n\n'
            + annotations['policy_uk'] + '\n\n'
            'Legacy збережена. 8 негативних validation-вирізок і всі зображення скопійовані байт-в-байт. '
            'Ця версія GT несумісна зі старими значеннями mAP; моделі потрібно оцінити заново. '
            'Validation містить дві відомі послідовності, це не новий незалежний test.\n\n'
            'Походження, старі/нові bbox, хеші та обмеження — у manifest.json.\n')
        a.out.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(a.out)
        print(json.dumps({k:v for k,v in manifest.items() if k != 'records'}, ensure_ascii=False, indent=2))
        print('Manifest SHA256:', sha(a.out / 'manifest.json'))
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == '__main__':
    main()
