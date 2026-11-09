#!/usr/bin/env python3
"""Звести виміряні варіанти моделей і явно позначити неперевірені гіпотези."""

import json
import hashlib
import math
import tempfile
from pathlib import Path
from common import ROOT, UkrainianParser
from report_io import write_json

R = ROOT / "reports"


def validate_comparability(profiles, allow_legacy=False):
    """Сувора перевірка протоколу; старі звіти без GT-хешів потребують явного винятку."""
    baseline = profiles[0]
    if baseline['status'] != 'виміряно':
        raise ValueError('Немає повного виміряного фото-орієнтира A.')
    legacy = False
    for profile in profiles:
        if profile['status'] != 'виміряно':
            continue
        for kind, report in profile['_reports'].items():
            expected = (dict(split='val', imgsz=640, conf=0.25, post_nms_iou=0.5, images=8)
                        if kind == 'background' else dict(split='val', device='cpu', imgsz=640, batch=4))
            for key, value in expected.items():
                if report.get(key) != value:
                    raise ValueError(f'{profile["id"]}/{kind}: несумісний {key}.')
            reference = baseline['_reports'][kind]
            for key in ['torch', 'ultralytics']:
                if report.get(key) != reference.get(key):
                    raise ValueError(f'{profile["id"]}/{kind}: різні версії {key}.')
            digest = report.get('dataset', {}).get('sha256')
            ref_digest = reference.get('dataset', {}).get('sha256')
            if digest and report['dataset'].get('images') != dict(video=32, cub=600, background=8)[kind]:
                raise ValueError(f'{profile["id"]}/{kind}: неочікувана кількість зображень.')
            evaluator = report.get('evaluator')
            ref_evaluator = reference.get('evaluator')
            if digest and ref_digest and digest != ref_digest:
                raise ValueError(f'{profile["id"]}/{kind}: різні дані або GT.')
            if evaluator and ref_evaluator:
                left = {k: v for k, v in evaluator.items() if k != 'source_class'}
                right = {k: v for k, v in ref_evaluator.items() if k != 'source_class'}
                if left != right:
                    raise ValueError(f'{profile["id"]}/{kind}: різні evaluators.')
            if not all([digest, ref_digest, evaluator, ref_evaluator]):
                legacy = True
                if not allow_legacy:
                    raise ValueError('Немає dataset/evaluator provenance. Для історичних звітів явно задайте --allow-legacy; їхню тотожність даних не можна довести з цих JSON.')
        for key in ['video_map50_95', 'cub_map50_95']:
            if not math.isfinite(profile[key]) or not 0 <= profile[key] <= 1:
                raise ValueError('Некоректна метрика mAP.')
        fp = profile['negative_images_with_fp']
        if not isinstance(fp, int) or not 0 <= fp <= 8:
            raise ValueError('Некоректна кількість фонових помилок.')
    return not legacy


def dominates(candidate, reference):
    return (candidate['video_map50_95'] >= reference['video_map50_95']
            and candidate['cub_map50_95'] >= reference['cub_map50_95']
            and candidate['negative_images_with_fp'] <= reference['negative_images_with_fp']
            and (candidate['video_map50_95'] > reference['video_map50_95']
                 or candidate['cub_map50_95'] > reference['cub_map50_95']))


def select_profile(profiles):
    base = profiles[0]
    if base['status'] != 'виміряно':
        raise ValueError('Для вибору потрібен виміряний A.')
    proposed = next(p for p in profiles if p['id'] == 'D')
    return 'D' if proposed['status'] == 'виміряно' and dominates(proposed, base) else 'A'


def read(name):
    return json.loads((R / name).read_text())


def measured(key, title, video, cub, bg, lr):
    if not all((R / f).exists() for f in [video, cub, bg]):
        return {
            "id": key,
            "name": title,
            "status": "виконується",
            "lr0": lr,
            "cub_train_images": 128,
        }
    v, c, b = read(video), read(cub), read(bg)
    if not v["weights_sha256"] == c["weights_sha256"] == b["weights_sha256"]:
        raise ValueError("Хеші варіанта не збігаються.")
    return {
        "id": key,
        "name": title,
        "status": "виміряно",
        "weights_sha256": v["weights_sha256"],
        "lr0": lr,
        "cub_train_images": 128,
        "video_map50_95": v["metrics"]["metrics/mAP50-95(B)"],
        "cub_map50_95": c["metrics"]["metrics/mAP50-95(B)"],
        "negative_images_with_fp": b["images_with_false_positives"],
        "sources": [video, cub, bg],
        "_reports": dict(video=v, cub=c, background=b),
    }


def main():
    parser = UkrainianParser(description=__doc__)
    parser.add_argument('--allow-legacy', action='store_true', help='Явно дозволити історичні JSON без хешів GT, із попередженням про неповну перевірку')
    parser.add_argument('--out-dir', type=Path, help='Новий каталог для звітів; типово runs/profile-comparison-*')
    args = parser.parse_args()
    profiles = [
        measured(
            "A",
            "Фото-орієнтир",
            "video_training_baseline.json",
            "expanded_candidate_validation.json",
            "video_backgrounds_baseline.json",
            None,
        ),
        measured(
            "B",
            "Відеопілот",
            "video_training_candidate.json",
            "video_training_cub_validation.json",
            "video_backgrounds_candidate.json",
            1e-5,
        ),
        measured(
            "C",
            "Відео: менший LR",
            "profile_c_video_validation.json",
            "profile_c_cub_validation.json",
            "profile_c_backgrounds.json",
            3e-6,
        ),
        measured(
            "D",
            "Більша частка фото",
            "profile_d_video_validation.json",
            "profile_d_cub_validation.json",
            "profile_d_backgrounds.json",
            1e-5,
        ),
    ]
    profiles[3]["cub_train_images"] = 512
    profiles[0]["cub_train_images"] = 5377
    profiles[0]["video_train_images"] = 0
    for item in profiles[1:]:
        item["video_train_images"] = 87
    try:
        verified = validate_comparability(profiles, allow_legacy=args.allow_legacy)
    except ValueError as error:
        parser.error(str(error))
    selected = select_profile(profiles)
    base = profiles[0]
    for p in profiles:
        if p["status"] == "виміряно":
            p["video_delta_pp"] = 100 * (p["video_map50_95"] - base["video_map50_95"])
            p["cub_delta_pp"] = 100 * (p["cub_map50_95"] - base["cub_map50_95"])
            p["dominates_release"] = dominates(p, base)
    for item in profiles:
        if item["status"] != "виміряно":
            item["decision"] = item["status"]
            continue
        dominated = dominates(base, item)
        item["decision"] = (
            "основний"
            if item["id"] == selected and selected != 'A'
            else (
                "фото-орієнтир"
                if item["id"] == "A"
                else ("відхилено" if dominated else "попередній пілот")
            )
        )
        item["published_weights"] = None
        for file in ["weights/best.pt", "weights/photo_reference.pt", "weights/video_pilot.pt"]:
            path = ROOT / file
            if (
                path.exists()
                and hashlib.sha256(path.read_bytes()).hexdigest() == item["weights_sha256"]
            ):
                item["published_weights"] = file
    result = {
        "default_profile": selected,
        "dataset_provenance_verified": verified,
        "provenance_note": ('Хеші даних та evaluators зіставлено.' if verified else 'Історичні звіти: тотожність GT/evaluator не підтверджена хешами; --allow-legacy задано явно.'),
        "reference_profile": "A",
        "datasets": {
            "cub_validation_images": 600,
            "video_validation_images": 32,
            "video_validation_positive_images": 24,
            "video_validation_background_crops": 8,
            "imgsz": 640,
            "device": "cpu",
            "batch": 4,
        },
        "rule": "Заміна релізної моделі — лише якщо обидві mAP50–95 не нижчі, одна вища, а хибних фонових спрацьовувань не більше. Компромісний варіант публікується окремо.",
        "profiles": profiles,
    }
    for profile in profiles:
        profile.pop('_reports', None)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=False)
        output = args.out_dir
    else:
        (ROOT / 'runs').mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix='profile-comparison-', dir=ROOT / 'runs'))
    write_json(output / 'model_profiles.json', result)

    def pct(v):
        return f"{100*v:.2f}%".replace(".", ",")

    s = """# Варіанти рішення

Варіанти мають різні цілі. Основна модель і відеокандидати порівнюються на **тих самих** 600 CUB- та 32 відеовалідаційних зображеннях: CPU, 640 px, batch=4.

| Варіант | Рішення / статус | CUB mAP50–95 | Відео mAP50–95 | Негативні кадри з FP / 8 |
|---|---|---:|---:|---:|
"""
    for p in profiles:
        if p["status"] == "виміряно":
            s += f'| {p["id"]}: {p["name"]} | {p["decision"]} | {pct(p["cub_map50_95"])} | {pct(p["video_map50_95"])} | {p["negative_images_with_fp"]} |\n'
        else:
            s += f'| {p["id"]}: {p["name"]} | **{p["status"]}** | — | — | — |\n'
    s += """
## Що відрізняється

- **A — фото-орієнтир:** попередні ваги `weights/photo_reference.pt`.
- **B — відеопілот:** 87 відеоприкладів + 128 фото CUB, 3 епохи, AdamW 1e-5; навчається вся голова.
- **C — менший крок:** ті самі дані, seed=26 і 3 епохи, LR=3e-6. Від B змінено лише LR.
- **D — баланс даних:** 87 відеоприкладів + 512 фото, LR=1e-5. Запущено окремо; змінює також кількість кроків за епоху, тому це окремий рецепт.

B і C починаються з однакових ваг A. Для них збігаються навчальні списки, seed і кількість епох; змінено лише LR.

## Поточний вибір

**D — основний варіант:** перевищив A на обох валідаціях, фонові помилки не зросли. **A і B залишені для порівняння. C відхилено в цьому запуску.**

```bash
# D: поточна основна модель.
python predict_video.py --input examples/input.mp4 --output examples/result_d.mp4 --weights weights/best.pt
# A: фото-орієнтир.
python predict_video.py --input examples/input.mp4 --output examples/result_a.mp4 --weights weights/photo_reference.pt
# B: попередній відеопілот.
python predict_video.py --input examples/input.mp4 --output examples/result_b.mp4 --weights weights/video_pilot.pt
```

Приклади: [D](../examples/output.mp4) · [A](../examples/output_photo_reference.mp4) · [B](../examples/output_video_pilot.mp4).

Історичні test-результати релізу наведені окремо в [основному звіті](../../reports/improvement.md). Цей скрипт не використовує test для вибору й не приписує його оцінки новим кандидатам. Малі відмінності одного запуску не доводять статистичної переваги.

## Правило вибору

Реліз замінюється лише коли обидві mAP50–95 не нижчі за A, хоча б одна вища, а фонових помилок не більше. Компроміс між фото й відео позначається окремо; він не називається універсальним покращенням.

Відеовалідація мала: 24 позитивні кадри та 8 перевірених фонових вирізок із двох роликів. Результат описує цей набір, не всі відеосцени. Test для вибору варіанта не використовується.

## Відтворення

Після підготовки відеоданих та CUB:

```bash
python train_video.py --model weights/photo_reference.pt --lr0 0.00001 --cub-images 128 --name video-head-pilot
python train_video.py --model weights/photo_reference.pt --lr0 0.000003 --cub-images 128 --name video-low-lr
# D: окремий рецепт із більшою часткою фото.
python train_video.py --model weights/photo_reference.pt --lr0 0.00001 --cub-images 512 --name video-more-photos
python compare_profiles.py
```

`compare_profiles.py` читає збережені JSON, не запускає навчання. Існуючі каталоги запусків не перезаписуються. Для іншого повтору задайте новий --name.

Точні числа, хеші та джерела — [model_profiles.json](model_profiles.json). Основний звіт — [improvement.md](improvement.md).
"""
    s = s.replace(
        "{{D_STATUS}}",
        (
            "D виконується; метрики ще не отримані."
            if profiles[3]["status"] != "виміряно"
            else "D оцінено; його результати й статус наведені в таблиці."
        ),
    )
    if selected != 'D':
        s = s.replace('**D — основний варіант:** перевищив A на обох валідаціях, фонові помилки не зросли. **A і B залишені для порівняння. C відхилено в цьому запуску.**',
                      '**Рекомендовано залишити A:** D не пройшов правило або ще не виміряний. Опубліковані ваги автоматично не змінюються.')
        s = s.replace('# D: поточна основна модель.', '# Опубліковані ваги D; у цьому порівнянні заміна A не рекомендована.')
    if not verified:
        s = s.replace('на **тих самих** 600 CUB- та 32 відеовалідаційних зображеннях',
                      'за історично заявленими вибірками 600 CUB / 32 відеоприклади; тотожність GT не підтверджена хешами')
        s = s.replace('\n## Що відрізняється', '\n**Увага:** --allow-legacy дозволяє лише історичне зіставлення. Дані та evaluator не мають повної збереженої provenance.\n\n## Що відрізняється')
    s = s.replace('python compare_profiles.py\n', 'python compare_profiles.py --allow-legacy\n')
    s = s.replace('(../examples/', '(https://github.com/tarasenkoandriii/bird-detector/blob/main/examples/')
    s = s.replace('(../../reports/improvement.md)', '(https://github.com/tarasenkoandriii/bird-detector/blob/main/reports/improvement.md)')
    s = s.replace('(improvement.md)', '(https://github.com/tarasenkoandriii/bird-detector/blob/main/reports/improvement.md)')
    (output / 'model_profiles.md').write_text(s)
    print('Звіти:', output)


if __name__ == "__main__":
    main()
