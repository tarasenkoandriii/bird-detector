# Детектор птахів — YOLO26n

Один клас `bird`. Скрипт обробляє кожен кадр відео, додає рамки, назву й впевненість та зберігає MP4.

![Приклад детекції](examples/frame-000.jpg)

[Вхідне відео](examples/input.mp4) · [Результат](examples/output.mp4) · [Підсумковий звіт](reports/improvement.md)

## Встановлення

Перевірено на стенді з конфігурацією: **Intel Core i7-8750H (6 ядер / 12 потоків), Radeon Pro 555X (4 ГБ), 32 ГБ RAM, macOS 15.8.1, Python 3.11.6**. Версії основних бібліотек зафіксовані; для Pillow і PyArrow задано допустимі діапазони в [requirements.txt](requirements.txt).

```bash
git clone https://github.com/tarasenkoandriii/bird-detector.git
cd bird-detector
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

Для H.264 рекомендовано FFmpeg: `brew install ffmpeg`. Без нього відео зберігається як `mp4v`, сумісність із браузером залежить від браузера.

## Обробка відео

Датасет завантажувати не потрібно.

```bash
python predict_video.py --input examples/input.mp4 --output examples/result.mp4
```

Типові параметри: `--weights weights/best.pt --imgsz 640 --conf 0.25 --iou 0.5 --device auto`. Пристрій можна задати явно: `cpu`, `mps` або індекс CUDA. Для заміни вже наявного вихідного файлу додайте `--overwrite`.

Зберігаються всі кадри, розмір і FPS; аудіо не копіюється. Вихід перевіряється повторним декодуванням. Поруч створюється `*.metrics.json` із часом, кількістю кадрів і хешем ваг.

## Датасет

[CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/): 200 видів об’єднані в один клас. Фінальний набір: **5 377 train / 600 validation / 499 test**. Вихідні bbox перетворені на нормалізований YOLO-формат; розмітку, дублікати й перетини вибірок перевірено. Перцептивний пошук схожих кадрів евристичний.

Дані завантажуються окремо, без ключів. Лише некомерційні дослідження та навчання; джерела й ліцензії — [THIRD_PARTY.md](THIRD_PARTY.md).

Для фінального донавчання D додано **87 відеоприкладів LaSOT і 512 фото CUB**. Окрема відеовалідація — **32 приклади: 24 позитивні кадри та 8 фонових вирізок**. Навчальні й валідаційні відеопослідовності розділені. [Підготовка, розмітка й аудит](reports/video_training.md).

## Навчання

Команди для чистої копії проєкту:

```bash
# Початкова підвибірка: 998/200/100.
python prepare_data.py
python train.py
python train_boxes.py --model runs/train/bird/weights/best.pt --name bbox-conservative

# Розширення зі збереженням захисту попередніх вибірок.
python prepare_expanded.py
python train_boxes.py --data datasets/bird-expanded/data.yaml \
  --model runs/experiments/bbox-conservative/weights/best.pt \
  --imgsz 640 --epochs 3 --lr0 0.00001 --name expanded-bbox640

# D: відеокадри + 512 фото CUB.
python prepare_video_training.py
python train_video.py --model runs/experiments/expanded-bbox640/weights/best.pt \
  --cub-images 512 --lr0 0.00001 --name video-more-photos
```

Етапи: 5 епох класифікаційних гілок голови детектора → 3 епохи bbox-гілок при 320 px → 3 епохи bbox-гілок при 640 px → 3 епохи всієї голови на суміші відео й фото. AdamW, batch=8, CPU; backbone і neck зафіксовані. Останній етап із валідаціями зайняв **8 хв 28 с**. Параметри — [training_provenance.json](reports/training_provenance.json).

Результат: `runs/experiments/video-more-photos/weights/best.pt`. Обробка відео власними навченими вагами:

```bash
python predict_video.py --input examples/input.mp4 --output examples/my_output.mp4 \
  --weights runs/experiments/video-more-photos/weights/best.pt
```

Додані `weights/best.pt` не перезаписуються. Для повторного навчання задайте нове `--name`; уже підготовлені каталоги даних пропустіть. Повторний запуск може дати дещо інші числа.

## Метрики на валідації та тесті

Метрики D на **CUB**: CPU, 640 px, batch=4. Кандидата обрано за фото- й відеовалідаціями; test оцінено після вибору.

| Метрика | Валідація, 600 | Тест, 499 |
|---|---:|---:|
| Precision | 95,96% | 97,87% |
| Recall | 94,98% | 96,99% |
| mAP50 | 98,19% | 99,17% |
| mAP50–95 | **81,17%** | **83,60%** |

Це поточний варіант **D**. Порівняно з A: CUB validation +0,09 в.п., нова відеовалідація +1,02 в.п.; CUB test −0,02 в.п. [Усі варіанти й правило вибору](reports/model_profiles.md).

```bash
python evaluate.py --data datasets/bird-expanded/data.yaml --split val --imgsz 640 \
  --report runs/evaluate/recheck-val.json
python evaluate.py --data datasets/bird-expanded/data.yaml --split test --imgsz 640 \
  --report runs/evaluate/recheck-test.json
python -m unittest discover -s tests -v
```

**37 тестів пройшли**, включно з перевірками даних, відео, статистики, заморожування шарів і tiled inference. Відеоприклад D: 240/240 кадрів, приблизно 10 кадрів/с на MPS. Це швидкість обробки, не FPS відтворення чи оцінка recall.

## Відеоперевірка

Окремий LaSOT-тест: **64,00% цільового recall при IoU≥0,5 на 300 кадрах**. Розмітка охоплює одного птаха на кадр, тому це не повний mAP відеодетектора. [Метрики й команди](reports/video_test.md).

## Варіанти моделі

| Варіант | Ваги | CUB / відео mAP50–95 на валідації |
|---|---|---:|
| [**D — основний**](examples/output.mp4) | [best.pt](weights/best.pt) | 81,17% / 51,98% |
| [A — фото-орієнтир](examples/output_photo_reference.mp4) | [photo_reference.pt](weights/photo_reference.pt) | 81,08% / 50,97% |
| [B — попередній відеопілот](examples/output_video_pilot.mp4) | [video_pilot.pt](weights/video_pilot.pt) | 80,81% / 51,94% |

За замовчуванням використовується D. Для іншого варіанта передайте його файл через `--weights`. C з меншим LR перевірено й відхилено. [Порівняння A/B/C/D](reports/model_profiles.md).

Стійкість фінального етапу D перевірена від однакових ваг A на фіксованих даних: **5/5 seed** пройшли критерій проти A. Середні mAP50–95 — **81,22% CUB / 51,51% відео**; вибіркове SD — **0,04 / 0,38 в.п.** Це не довірчі інтервали. [Протокол і результати](reports/seed_study/README.md).

## Дослідження стенда

На однакових даних і параметрах прогріта train-епоха CPU — **24,98 с**, MPS — **27,36 с**. Це один короткий парний замір, а не загальна перевага CPU. Якість обох запусків оцінена на CPU; релізні ваги не змінювалися. [Протокол, швидкість і пам’ять](reports/stand_benchmark/README.md).

## Проблеми та можливі покращення

- Відеовалідація мала: 24 позитивні кадри й 8 фонових вирізок. Результати одного запуску не доводять статистичної переваги на всіх відеосценах.
- У перевіреному стеку torch 2.2.2/MPS виявлено спотворення відновлення координат: `SafeDetectionPredictor` виконує цей крок на CPU, залишаючи нейромережу на вибраному пристрої.
- Відео має додатковий NMS IoU=0.5; офіційні метрики валідатора його не включають. Покадрової еталонної розмітки прикладу немає.

[Підготовка й запуск у Warp](reports/expanded_dataset.md) · [Рекомендації з донавчання](reports/finetuning_recommendations.md) · [Перевірки](reports/verification.md)

Код — AGPL-3.0. Умови датасету й відео наведені окремо в THIRD_PARTY.md.

## Перспективні напрями

Окремою короткою серією перевірено повніші відеодані, head/neck, YOLO26s і перекривні фрагменти кадру. [Протокол і результати](reports/video_study/README.md). Подальші напрями:

1. Різноманітніші відеосцени з повною розміткою всіх птахів і негативних кадрів.
2. SAHI — детекція дрібних птахів на перекривних фрагментах кадру, з контролем FPS.
3. Часткове розморожування neck для адаптації ознак до відео.
4. Дистиляція YOLO26s → YOLO26n зі збереженням nano для інференсу.
5. P2-голова — окрема архітектурна гілка для дрібних об’єктів.

[Обґрунтування, джерела та обмеження](reports/major_improvements.md).

## Додаткова відеоадаптація

На **v2-розмітці всього видимого птаха** D має 77,97% mAP50–95; після 3 епох head-only — **78,50%** (+0,53 в.п.). На CUB validation — **81,08%** проти D **81,17%**: це компроміс, а не одночасне покращення всіх метрик. 5 епох і neck не перевершили цей рецепт; YOLO26s після 3 епох отримала 71,44%.

Експериментальні ваги збережено окремо — [video_focus.pt](weights/video_focus.pt). `weights/best.pt` і порівняння A–D залишаються незмінними. v2 використовує ті самі відомі validation-кадри з уточненими рамками; її числа не порівнюються з legacy-метриками 50–52% вище. Це development-перевірка, не новий незалежний test.

```bash
python predict_video.py --input examples/input.mp4 --output examples/my_video_focus.mp4 \
  --weights weights/video_focus.pt --device cpu
```

[Відеоприклад експериментальних ваг](examples/output_video_focus.mp4) · [Усі запуски, дані та команди відтворення](reports/video_study/README.md).

## Статус

**Основне ТЗ виконано:** A–D порівняні, стійкість D перевірена на 5 seed, CPU/MPS benchmark виконаний. Код, ваги, метрики й відеоприклади готові для здачі. Додаткова коротка серія відеоадаптації описана окремо; її один seed не замінює дослідження стійкості D.

## Файли для здачі

[Код і README](https://github.com/tarasenkoandriii/bird-detector) · [Навчені ваги](weights/best.pt) · [Вхідне відео](examples/input.mp4) · [Оброблене відео](examples/output.mp4).

```text
bird-detector/
├── README.md
├── requirements.txt
├── train.py
├── predict_video.py
├── weights/best.pt
└── examples/
    ├── input.mp4
    └── output.mp4
```

Додаткові скрипти підготовки, навчання й оцінювання забезпечують відтворення описаних результатів.


## Презентація

- [Редагована презентація PPTX](Bird_Detector_Presentation_UK.pptx) — 60 слайдів із нотатками диктора.
- [Статична версія PDF](Bird_Detector_Presentation_UK.pdf).
- [Відео з українським озвученням](Bird_Detector_Presentation_UK.mp4) — голос Daniel, Soniox `tts-rt-v2`, швидкість 1,15×.

Повний комплект окремих кліпів, загальної аудіодоріжки, субтитрів, графіки та звіту аудиту доступний в архіві комплекту. Gemini перевірив аудіо слайдів 1–30; слайди 31–60 не перевірені, а виявлені помилки не для всіх слайдів повторно синтезовані. Деталі — у `Аудит_озвучення_UK.md` в архіві комплекту.
