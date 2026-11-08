# Донавчання на розширеному CUB

Поточний реліз: YOLO26n, один клас bird. Навчання завершено 8 листопада 2026 року. SHA256 `weights/best.pt`: `e0085bc7dfbb260d2411d72d42d15d66d65156a32b3b051943515b695fe4c276`.

## Результат на однакових вибірках

| Показник при 640, CPU | До expanded-етапу | Після expanded-етапу | Зміна, в.п. |
|---|---:|---:|---:|
| Validation mAP50–95, 600 зображень | 80,24% | **81,08%** | +0,84 |
| Test mAP50–95, 499 зображень | 82,88% | **83,62%** | +0,73 |
| Test mAP50 | 99,11% | **99,16%** | +0,05 |
| Test Precision | 98,00% | **98,21%** | +0,20 |
| Test Recall | 96,39% | **96,59%** | +0,20 |

Batch=4, conf=0.001, IoU=0.7. Validation Precision=95,28%, Recall=95,17%, mAP50=98,06%. Основний приріст — точність рамок; приріст пошуку птахів невеликий. Відсоткові пункти обчислено з точних чисел до округлення.

Baseline і кандидат оцінені на тих самих розширених split. Результати на попередніх 200/100 зображеннях не використовуються як безпосереднє порівняння. Попередній bbox-етап описано окремо в [bbox_stage_improvement.md](bbox_stage_improvement.md).

## Протокол і вибір кандидата

- Дані: **5 377 train / 600 validation / 499 резервних test**, 200 species CUB об’єднано в bird.
- Попередні validation/test виключені з train; підготовка й аудит — [expanded_dataset.md](expanded_dataset.md).
- Стартові ваги: попередній bbox-checkpoint із SHA256 `f61344a84ebc357103459dd98ada79d80abbcccdbd8f1d9c6e1ba94bcb140d18`.
- CPU, 3 епохи, imgsz=640, batch=8, nbs=64, AdamW lr0=0.00001, lrf=0.1, weight_decay=0, seed=26.
- Навчаються bbox-гілки cv2 та one2one_cv2; backbone, neck і класифікація зафіксовані. Статистика всіх BN зафіксована; affine-параметри bbox-BN можуть навчатися.
- Mosaic/scale/translate/fliplr вимкнені, warmup=0; типові HSV-аугментації залишаються активними.
- Захищені параметри перевірено після кожної епохи: незмінні.
- Найкращий checkpoint — третя епоха, обрана за validation fitness. Після тренера його окремо оцінено з batch=4.
- Рішення про кандидата прийнято за validation. Лише після цього baseline і кандидат оцінені на резервному test; наступного підбору за test не проводилося. Тепер цей test уже розкритий і не повинен використовуватися для подальшого пошуку параметрів.

Повна робота з фінальними оцінками тривала **47 хв 59 с**. Власне train-частини епох: 849,08 / 829,20 / 866,24 с; epoch validation: 72,65 / 70,80 / 71,11 с. Припущення benchmark про 63 хв було консервативним; тривалий запуск виявився швидшим.

## Відео та перевірки

На CPU та MPS окремо оброблено всі 240 кадрів прикладу при 640 без пропусків: 245 рамок, 240 кадрів із детекціями, 23,93 с обробки — близько 10,03 кадрів/с. FPS відтворення, розмір і тривалість збережені; MP4 і контрольні кадри оновлені. Усі 17 автоматичних тестів пройшли після заміни ваг.

Відео має додатковий NMS IoU=0.5. Офіційні числові метрики валідатора вище не включають цей крок. Покадрової GT-розмітки відео немає: кількість рамок не є precision або recall.

CUB — фотографії птахів; високі показники не доводять таку саму якість для далеких птахів у польоті, негативних відеосцен чи незнайомих камер. Можливі перетини CUB із даними pretraining. Новий test був незалежним від локального fine-tuning, але не гарантовано від попереднього навчання.

## Відтворення

Після підготовки розширеного набору:

```bash
python train_boxes.py --data datasets/bird-expanded/data.yaml \
  --model runs/experiments/bbox-conservative/weights/best.pt \
  --imgsz 640 --epochs 3 --lr0 0.00001 --name expanded-bbox640
python evaluate.py --data datasets/bird-expanded/data.yaml --split val --imgsz 640
python evaluate.py --data datasets/bird-expanded/data.yaml --split test --imgsz 640
python predict_video.py --input examples/input.mp4 --output examples/output.mp4 --overwrite
python -m unittest discover -s tests -v
```

Перші два етапи для отримання стартових ваг наведені в README. Для оцінки власного checkpoint передайте `--weights`; train-скрипти не перезаписують додані ваги автоматично. Наявні каталоги запусків не перезаписуються: задавайте нове --name. Повторне навчання може дати дещо інші числа.

Точні звіти: [expanded_baseline.json](expanded_baseline.json), [expanded_candidate_validation.json](expanded_candidate_validation.json), [expanded_test_baseline.json](expanded_test_baseline.json), [expanded_test_candidate.json](expanded_test_candidate.json). Історія, параметри та перевірки: expanded_training_history.csv, expanded_training_protocol.json, expanded_training_timing.json, expanded_protected_parameters.json, expanded_training_result.json.
