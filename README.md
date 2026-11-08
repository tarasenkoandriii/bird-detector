# Bird Detector — YOLO26n

Детекція одного класу `bird` на зображеннях і **кожному кадрі відео**. У репозиторії є навчені ваги, справжній вхідний MP4, оброблений MP4 та приклади кадрів.

**Оновлення моделі:** виконано двоетапне донавчання класифікації та bbox-гілок. Порівняння з початковою YOLO26n та аналіз помилок наведені в [звіті про покращення](reports/improvement.md).

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

Типовий рекомендований запуск: pretrained `yolo26n.pt`, **5 епох на CPU**, imgsz=320, batch=8, AdamW lr0=0.0001, warmup=0, seed=26. Навчаються лише `cv3` та `one2one_cv3` останньої голови: backbone, neck і bbox-гілки зафіксовані. Після кожної епохи перевіряється, що захищені параметри не змінилися. На перевіреному Intel Mac короткий експеримент тривав приблизно 10 хвилин.

```bash
python train.py
python train.py --epochs 5 --batch 8 --imgsz 320 --device cpu --strategy head
python train.py --epochs 1 --batch 2 --device cpu --name smoke
```

Початкові pretrained ваги за потреби завантажуються Ultralytics. Для повторного запуску задайте нове `--name`, якщо каталог уже існує. Детальний експеримент можна відтворити також через `python train_head.py`.

Нові результати — `runs/train/bird/weights/best.pt`, `results.csv`, `protocol.json`, `protected_parameters.json`. Доданий `weights/best.pt` автоматично не перезаписується. Для нового відео передайте `--weights` із потрібним шляхом.

Відновлення незавершеного навчання:

```bash
python train.py --resume runs/train/bird/weights/last.pt --strategy head --batch 8 --device cpu
```

Фінальні ваги після завершення мають видалений optimizer і не призначені для `--resume`. Для нового fine-tuning використовуйте `--model weights/best.pt` без `--resume`.

Повне донавчання всіх параметрів доступне як окремий, **не рекомендований за замовчуванням** експеримент:

```bash
python train.py --strategy full --epochs 50 --batch 8 --lr0 0.001 --name full-experiment
```

Поточні `weights/best.pt` отримані за два етапи: 5 епох класифікації та 3 епохи bbox-гілок (обрано checkpoint першої bbox-епохи); параметри — [training_provenance.json](reports/training_provenance.json). Новий запуск не гарантує побітового збігу ваг.

## Метрики

Еталонний повторний запуск **тих самих best.pt** на CPU, imgsz=320, batch=4. Validation і test розділені; test не використовується для підбору confidence.

| Метрика | Validation, 200 зображень | Test, 100 зображень |
| --- | ---: | ---: |
| Precision | 95.95% | 96.84% |
| Recall | 95.00% | 91.81% |
| mAP50 | 97.05% | 95.36% |
| mAP50–95 | 79.26% | 78.96% |

Повні числа, версії, SHA256 і швидкість — [val.json](reports/val.json) та [test.json](reports/test.json). Перевірити:

```bash
python evaluate.py --split val --device cpu
python evaluate.py --split test --device cpu
```

Для відеоприкладу використано 640 px, а не 320 px оцінки — це явно інша конфігурація інференсу.

## Проблеми, обмеження та покращення

- **Старий MPS і масштабування bbox.** На torch 2.2.2 прямокутні кадри давали різні координати на CPU та MPS. `SafeDetectionPredictor` переносить raw detections на CPU перед відновленням координат. Нейромережа залишається на вибраному пристрої; встановлені пакети не змінюються. Регресійний тест перевіряє масштаб і padding. Для еталонних метрик використано CPU.
- **Перенесення на інші сцени.** Нові види фону, далекі птахи та польоти потребують окремої перевірки. Це дослідницька модель, а не гарантія промислової якості.
- **Повторні рамки.** Для відео застосовано додатковий CPU NMS, IoU=0.5. На щільних сценах його поріг може впливати на близько розташованих птахів.
- **Зсув домену.** CUB — фотографії окремих птахів, не перевірка далеких птахів у польоті чи всіх птахів у багатопташиній сцені. Спеціальних негативних відеосцен у підвибірці немає; можливі перетини з даними попереднього навчання.
- **MPS fallback.** Непідтримувані операції, зокрема NMS у цьому стеку, можуть виконуватися на CPU. Сам факт GPU-доступності не доводить прискорення. При OOM зменшіть batch або використайте CPU.
- **Наступні покращення:** відеокадри з ручною розміткою, негативні приклади, візуальний аналіз FP/FN, контрольований експеримент з більшим train imgsz та порогом confidence на validation. Трекінг не є частиною цього мінімального завдання.

## Порівняння моделей та аналіз помилок

```bash
python compare_models.py
python analyze_errors.py
```

Порівняння виконується лише на validation. У початковій COCO-моделі bird має індекс 14: перед оцінкою відбирається саме цей клас і перетворюється на 0. Інші COCO-класи не враховуються як птахи. Аналіз помилок використовує явно зафіксовані conf=0.25 та IoU=0.5; його Precision/Recall не тотожні оптимальній точці PR-кривої в таблиці вище.

## Перевірки

```bash
python -m unittest discover -s tests -v
```

Тести покривають збереження кожного кадру, порожні детекції, відмову перезапису/однакового input-output, очищення часткового файлу після помилки, некоректне відео, NMS, валідацію розмітки та cross-split дублікати. Є CPU/MPS-регресія координат (MPS-гілка виконується за доступності GPU).

Підтверджено встановлення у новий venv, `pip check`, коротке справжнє навчання з backward/checkpoint, повторну оцінку best.pt на validation/test і обробку реального MP4. Детальний протокол — [verification.md](reports/verification.md). У поточній версії 16 автоматичних тестів, включно з COCO→bird mapping, вибором навчальних параметрів і обліком дублікатів як FP.

## Структура

```text
bird-detector/
├── README.md
├── requirements.txt
├── train.py
├── train_head.py
├── compare_models.py
├── analyze_errors.py
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

Репозиторій знову публічний за запитом власника. Оригінальний текст GNU AGPL у LICENSE збережено без перекладу; українське пояснення джерел і ліцензій — THIRD_PARTY.md.

## Додаткові експерименти з якістю

Перевірено роздільності 448–1280 та об’єднання звичайного й дзеркального прогнозів. 512 покращив validation, але перевага не підтвердилася на test; типовий режим 640 збережено. Протокол, метрики та команди — [further_experiments.md](reports/further_experiments.md). Ваги не змінено.

## Донавчання точності рамок

Після класифікаційного етапу запустіть:

```bash
python train_boxes.py --model runs/train/bird/weights/best.pt --name bbox-conservative
```

Скрипт навчає лише bbox-гілки 3 епохи на CPU із lr0=0.00001; backbone, neck, класифікація та статистика BN зафіксовані. Результат — `runs/experiments/bbox-conservative/weights/best.pt`. Додані ваги автоматично не перезаписуються. На цьому Mac етап зайняв близько 4 хвилин. При 640 validation mAP50–95 = **81,41%**, test = **80,80%**; для обох використано CPU. Це метрики офіційного валідатора; додатковий відео-NMS є окремою постобробкою. Деталі — [improvement.md](reports/improvement.md).

Огляд офіційних рекомендацій, перевірка локального API та рецепти наступних експериментів — [finetuning_recommendations.md](reports/finetuning_recommendations.md). Це план, поточні ваги ним не змінені.
