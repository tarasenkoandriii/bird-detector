# Bird Detector — YOLO26n

Детекція одного класу `bird` на зображеннях і **кожному кадрі відео**. У репозиторії є навчені ваги, справжній вхідний MP4, оброблений MP4 та приклади кадрів.

![Приклад детекції](examples/frame-000.jpg)

[Вхідне відео](examples/input.mp4) · [Відео з детекціями](examples/output.mp4) · [Метрики validation](reports/val.json)

## Встановлення

Перевірено: Python **3.11.6**, macOS 15.8.1, Intel i7/Radeon Pro 555X, 32 GB RAM. `torch==2.2.2` і `torchvision==0.17.2` зафіксовані як сумісна пара; NumPy залишається на 1.26.4. Інші платформи не перевірялися окремо.

```bash
git clone https://github.com/tarasenkoandriii/bird-detector.git
cd bird-detector
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

FFmpeg необов’язковий для інференсу, але рекомендований для MP4/H.264, що відтворюється в браузері та QuickTime. На macOS за наявності Homebrew: `brew install ffmpeg`. Якщо FFmpeg відсутній, скрипт збереже MPEG-4 Part 2 (`mp4v`); сумісність такого файлу з браузером залежить від браузера. Додані в репозиторій MP4 вже мають H.264.

## Обробка відео готовою моделлю

Датасет для цього **не потрібно завантажувати**:

```bash
python predict_video.py --input examples/input.mp4 --output examples/output.mp4 --overwrite
```

`--overwrite` потрібен у цьому прикладі, тому що готовий `examples/output.mp4` вже включений. Для нового шляху достатньо `--input` та `--output`:

```bash
python predict_video.py --input examples/input.mp4 --output examples/my_result.mp4
python predict_video.py --input examples/input.mp4 --output examples/cpu_result.mp4 --device cpu
```

Типові параметри: `--weights weights/best.pt`, `--imgsz 640`, `--conf 0.25`, `--iou 0.5`, `--device auto`. Автоматичний режим обирає MPS, потім CUDA, інакше CPU. Можна явно задати `--device mps`, `--device cpu` або індекс CUDA-пристрою.

Кожен кадр отримує інференс; пропуск кадрів не застосовується. На виході — bbox, `bird` та confidence. FPS відтворення, розмір і кількість кадрів зберігаються. Для цього мінімального прикладу аудіодоріжка не копіюється; найкраще використовувати відео зі сталою частотою кадрів (CFR).

Незавершений файл зберігається тимчасово й публікується під кінцевим ім’ям лише після повторної перевірки всіх кадрів. Поруч створюється `*.metrics.json`: кількість кадрів, час, швидкість обробки, детекції та SHA256 ваг. Це технічний звіт, **не оцінка precision/recall відео**.

На прикладі 8 секунд, 640×360, 29.97 FPS: усі 240 кадрів збережені. Частота відтворення не є швидкістю обчислення; realtime не заявляється. Фактичні заміри поточного прикладу — [output.metrics.json](examples/output.metrics.json).

## Датасет і підготовка

Використано **CUB-200-2011**, публічна копія з bounding boxes:
https://huggingface.co/datasets/bentrevett/caltech-ucsd-birds-200-2011 . Оригінальне джерело та умови: https://www.vision.caltech.edu/datasets/cub_200_2011/ . **Зображення призначені для некомерційних досліджень і навчання.** Авторство, ліцензії прикладу та зміни описані в [THIRD_PARTY.md](THIRD_PARTY.md).

Підготовка не потребує Roboflow, токена або реєстрації. Потрібен інтернет і приблизно 564 MB завантаження; додаткове місце потрібне для кешу та підготовлених зображень.

```bash
python prepare_data.py
```

Скрипт завантажує перший із двох шардiв офіційного train та test, фіксує ревізію і перевіряє SHA256. Seed=26; до 1000 train і 200 validation вибираються з офіційного train, до 100 test — з офіційного test. Два приклади з bbox за межами зображення виключаються: підсумок **998 train / 200 validation / 100 test**, 100 представлених видів → один клас `0: bird`.

BBox джерела `[x0,y0,x1,y1]` перетворюються на нормалізовані YOLO `class x_center y_center width height`. Аудит перевіряє читабельність зображень, label-файли, класи, скінченність координат, межі bbox і дублікати. Групи дублікатів зберігають одного представника з пріоритетом test → validation → train. Near-hash евристичний; семантична відсутність leakage не доведена. На цій підвибірці аудит не знайшов помилок або пар дублікатів після виключення двох вихідних некоректних bbox.

Результати: `datasets/bird/data.yaml`, `audit.json`, `source.json`, `manifest.json`. Датасет не включений у Git. Існуючі каталоги не перезаписуються. Для повторної підготовки задайте новий `--out`; для власного YOLO export використайте `--source`.

## Навчання

Після підготовки даних:

```bash
python train.py
```

Типовий запуск: pretrained `yolo26n.pt`, 50 епох, imgsz=320, batch=8, workers=0, amp=False, AdamW lr0=0.001, mosaic=0, seed=26. Вихідні pretrained ваги за потреби завантажуються Ultralytics при першому запуску.

```bash
python train.py --data datasets/bird/data.yaml --epochs 50 --batch 8 --imgsz 320 --device mps
python train.py --epochs 1 --batch 2 --device cpu --name smoke
```

Нові результати — `runs/train/bird*/weights/best.pt` та `results.csv`; доданий `weights/best.pt` автоматично не замінюється. Для інференсу новими вагами передайте `--weights`.

Відновлення **незавершеного** навчання:

```bash
python train.py --resume runs/train/bird/weights/last.pt --batch 8 --device mps
```

Фінальні best.pt/last.pt після 50 епох мають видалений optimizer і не призначені для `--resume`. Для нового fine-tuning використовуйте `--model weights/best.pt` без `--resume`.

Додані ваги отримані під час 50-епохового експерименту з batch 2 → 4 → 6 → 8 → 10; точний графік — [training_provenance.json](reports/training_provenance.json). Seed задано, але новий типовий запуск не обіцяє побітового збігу з цими вагами. Час у сирому [training_history.csv](reports/training_history.csv) починається заново після відновлень. Best обирався за validation fitness; це не обов’язково остання епоха.

## Метрики

Еталонний повторний запуск **тих самих best.pt** на CPU, imgsz=320, batch=4. Validation і test розділені; test не використовується для підбору confidence.

| Метрика | Validation, 200 зображень | Test, 100 зображень |
| --- | ---: | ---: |
| Precision | 52.31% | 58.62% |
| Recall | 51.00% | 43.92% |
| mAP50 | 48.47% | 46.79% |
| mAP50–95 | 15.44% | 17.33% |

Повні числа, версії, SHA256 і швидкість — [val.json](reports/val.json) та [test.json](reports/test.json). Перевірити:

```bash
python evaluate.py --split val --device cpu
python evaluate.py --split test --device cpu
```

Ці метрики не слід змішувати з історичними MPS-числами. Останні залишені окремо в [historical_mps_test.json](reports/historical_mps_test.json). Для відеоприкладу використано 640 px, а не 320 px оцінки — це явно інша конфігурація інференсу.

## Проблеми, обмеження та покращення

- **Старий MPS і масштабування bbox.** На torch 2.2.2 прямокутні кадри давали різні координати на CPU та MPS. `SafeDetectionPredictor` переносить raw detections на CPU перед відновленням координат. Нейромережа залишається на вибраному пристрої; встановлені пакети не змінюються. Регресійний тест перевіряє масштаб і padding. Для еталонних метрик використано CPU.
- **Невисокий recall і точність локалізації.** Є пропуски, рамки можуть не охоплювати хвіст або всю птицю; модель не є production-ready. На цьому відео 640 px візуально працює краще за 320; це не гарантія для інших сцен.
- **Повторні рамки.** Для відео застосовано додатковий CPU NMS, IoU=0.5. На щільних сценах його поріг може впливати на близько розташованих птахів.
- **Зсув домену.** CUB — фотографії окремих птахів, не перевірка далеких птахів у польоті чи всіх птахів у багатопташиній сцені. Спеціальних негативних відеосцен у підвибірці немає; можливі перетини з даними попереднього навчання.
- **MPS fallback.** Непідтримувані операції, зокрема NMS у цьому стеку, можуть виконуватися на CPU. Сам факт GPU-доступності не доводить прискорення. При OOM зменшіть batch або використайте CPU.
- **Наступні покращення:** відеокадри з ручною розміткою, негативні приклади, візуальний аналіз FP/FN, контрольований експеримент з більшим train imgsz та порогом confidence на validation. Трекінг не є частиною цього мінімального завдання.

## Перевірки

```bash
python -m unittest discover -s tests -v
```

Тести покривають збереження кожного кадру, порожні детекції, відмову перезапису/однакового input-output, очищення часткового файлу після помилки, некоректне відео, NMS, валідацію розмітки та cross-split дублікати. Є CPU/MPS-регресія координат (MPS-гілка виконується за доступності GPU).

Підтверджено встановлення у новий venv, `pip check`, коротке справжнє навчання з backward/checkpoint, повторну оцінку best.pt на validation/test і обробку реального MP4. Детальний протокол — [verification.md](reports/verification.md).

## Структура

```text
bird-detector/
├── README.md
├── requirements.txt
├── train.py
├── predict_video.py
├── prepare_data.py
├── evaluate.py
├── common.py
├── inference.py
├── weights/best.pt
├── examples/{input.mp4,output.mp4,frame-*.jpg,output.metrics.json}
├── tools/                   # завантаження та аудит YOLO
├── tests/
└── reports/                 # метрики й історія експерименту
```

Відео — Matt MacGillivray, CC BY 2.0; джерело та всі зміни наведені в [THIRD_PARTY.md](THIRD_PARTY.md). Python-код — AGPL-3.0, умови сторонніх матеріалів окремі; див. [LICENSE](LICENSE).

Репозиторій приватний за запитом власника. Для перегляду та клонування потрібен дозволений GitHub-обліковий запис; перевіряльнику слід надати доступ. Оригінальний текст GNU AGPL у LICENSE збережено без перекладу; українське пояснення джерел і ліцензій — THIRD_PARTY.md.
