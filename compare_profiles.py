#!/usr/bin/env python3
"""Звести виміряні варіанти моделей і явно позначити неперевірені гіпотези."""

import json
import hashlib
from pathlib import Path
from common import ROOT

R = ROOT / "reports"


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
    }


def main():
    profiles = [
        measured(
            "A",
            "Релізна модель",
            "video_training_baseline.json",
            "val.json",
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
    base = profiles[0]
    for p in profiles:
        if p["status"] == "виміряно":
            p["video_delta_pp"] = 100 * (p["video_map50_95"] - base["video_map50_95"])
            p["cub_delta_pp"] = 100 * (p["cub_map50_95"] - base["cub_map50_95"])
            p["dominates_release"] = (
                p["video_map50_95"] >= base["video_map50_95"]
                and p["cub_map50_95"] >= base["cub_map50_95"]
                and p["negative_images_with_fp"] <= base["negative_images_with_fp"]
                and (
                    p["video_map50_95"] > base["video_map50_95"]
                    or p["cub_map50_95"] > base["cub_map50_95"]
                )
            )
    for item in profiles:
        if item["status"] != "виміряно":
            item["decision"] = item["status"]
            continue
        dominated = (
            base["video_map50_95"] >= item["video_map50_95"]
            and base["cub_map50_95"] >= item["cub_map50_95"]
            and base["negative_images_with_fp"] <= item["negative_images_with_fp"]
            and (
                base["video_map50_95"] > item["video_map50_95"]
                or base["cub_map50_95"] > item["cub_map50_95"]
            )
        )
        item["decision"] = (
            "основний"
            if item["id"] == "A"
            else ("відхилено" if dominated else "компромісний пілот")
        )
        item["published_weights"] = None
        for file in ["weights/best.pt", "weights/video_pilot.pt"]:
            path = ROOT / file
            if (
                path.exists()
                and hashlib.sha256(path.read_bytes()).hexdigest() == item["weights_sha256"]
            ):
                item["published_weights"] = file
    result = {
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
    (R / "model_profiles.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

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

- **A — реліз:** чинні `weights/best.pt`; основний орієнтир якості на CUB.
- **B — відеопілот:** 87 відеоприкладів + 128 фото CUB, 3 епохи, AdamW 1e-5; навчається вся голова.
- **C — менший крок:** ті самі дані, seed=26 і 3 епохи, LR=3e-6. Від B змінено лише LR.
- **D — баланс даних:** 87 відеоприкладів + 512 фото, LR=1e-5. Запущено окремо; змінює також кількість кроків за епоху, тому це окремий рецепт.

B і C починаються з однакових ваг A. Для них збігаються навчальні списки, seed і кількість епох; змінено лише LR.

## Поточний вибір

**A залишається основним. B доступний окремо як відеопілот:** на цій відеовалідації він кращий, на CUB — гірший. **C відхилено в цьому запуску:** він поступився A за обома mAP50–95. {{D_STATUS}}

```bash
# A: основна модель.
python predict_video.py --input examples/input.mp4 --output examples/photo_profile.mp4 --weights weights/best.pt
# B: експериментальний відеопілот.
python predict_video.py --input examples/input.mp4 --output examples/video_profile.mp4 --weights weights/video_pilot.pt
```

Приклади: [A](../examples/output.mp4) · [B](../examples/output_video_pilot.mp4).

Малі відмінності одного запуску не доводять статистичної переваги.

Test-метрики інших звітів стосуються A; для B/C тестові результати не заявляються.

## Правило вибору

Реліз замінюється лише коли обидві mAP50–95 не нижчі за A, хоча б одна вища, а фонових помилок не більше. Компроміс між фото й відео позначається окремо; він не називається універсальним покращенням.

Відеовалідація мала: 24 позитивні кадри та 8 перевірених фонових вирізок із двох роликів. Результат описує цей набір, не всі відеосцени. Test для вибору варіанта не використовується.

## Відтворення

Після підготовки відеоданих та CUB:

```bash
python train_video.py --lr0 0.00001 --cub-images 128 --name video-head-pilot
python train_video.py --lr0 0.000003 --cub-images 128 --name video-low-lr
# D: окремий рецепт із більшою часткою фото.
python train_video.py --lr0 0.00001 --cub-images 512 --name video-more-photos
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
    (R / "model_profiles.md").write_text(s)


if __name__ == "__main__":
    main()
