> Історичний bbox-етап до розширення набору. Поточні метрики — improvement.md.

# Покращення локалізації птахів

8 листопада 2026 року виконано контрольоване донавчання bbox-гілок після п’яти епох класифікаційного навчання. Дані незмінні: 998 train, 200 validation, 100 test.

## Результат

| Конфігурація | До bbox-етапу | Після bbox-етапу | Приріст, в.п. |
|---|---:|---:|---:|
| Validation mAP50–95, 320 | 77,58% | 79,26% | +1,68 |
| Validation mAP50–95, 640 | 79,46% | 81,41% | +1,95 |
| Test mAP50–95, 320 | 77,50% | 78,96% | +1,46 |
| Test mAP50–95, 640 | 79,48% | 80,80% | +1,32 |

Усі числа — CPU, batch=4, conf=0.001, IoU=0.7, однакові split. Виграш стосується точності рамок. При 640 test mAP50 = 98,65%, Precision = 99,63%, Recall = 94,00%; показники пошуку птахів практично незмінні. Невелика вибірка не дає підстав стверджувати статистичну значущість або переносити результат на довільні відеосцени.

## Протокол

- Початкова модель для bbox-етапу: класифікаційний checkpoint із SHA256 `95fcef6935b48f663c10dc4bfbd71d2bc8d11ca2ebef77ddefaa85e1419f995f`.
- 3 епохи, CPU, imgsz=320, batch=8, AdamW lr0=0.00001, lrf=0.1, weight_decay=0, seed=26.
- Навчаються лише cv2 та one2one_cv2 останньої голови. Backbone, neck і класифікація зафіксовані; статистика всіх BN зафіксована.
- Mosaic, scale, translate та fliplr вимкнені; warmup=0.
- Після кожної епохи перевірено незмінність захищених параметрів. Найкращий checkpoint — перша епоха за validation fitness. Повторна оцінка з batch=4 може відрізнятися від внутрішньої оцінки тренера з іншим batch.
- Рішення про застосування прийнято за validation 320/640. Test оцінено після вибору; наступного підбору параметрів за test не було. Test уже використовувався для попередніх звітів, тому нова незалежна вибірка потрібна для остаточної оцінки узагальнення.

Поточні `weights/best.pt`: SHA256 `f61344a84ebc357103459dd98ada79d80abbcccdbd8f1d9c6e1ba94bcb140d18`.

## Відтворення

```bash
python prepare_data.py
python train.py
python train_boxes.py --model runs/train/bird/weights/best.pt --name bbox-conservative
python compare_models.py --skip-base --trained runs/experiments/bbox-conservative/weights/best.pt
python evaluate.py --split test --weights runs/experiments/bbox-conservative/weights/best.pt --imgsz 640 --report reports/reproduced_bbox_stage_test.json
python predict_video.py --input examples/input.mp4 --output examples/output.mp4 --overwrite
python -m unittest discover -s tests -v
```

Для оцінки власних ваг передайте їх явно; train-скрипти не перезаписують додані weights/best.pt. Числовий результат повторного навчання може відрізнятися.

Точні результати: bbox_validation.json, bbox_stage_val.json, bbox_stage_test.json, bbox_stage_test_640.json; параметри й історія: bbox_training_protocol.json, bbox_training_history.csv, bbox_protected_parameters.json та training_provenance.json. Порівняння з початковою COCO YOLO26n — model_comparison.json; bird=14 відбирається перед перетворенням на bird=0.

Відео обробляється при 640 із додатковим NMS IoU=0.5. Офіційні метрики вище не включають цей додатковий крок. Кількість детекцій на демонстраційному відео не є recall — ручної покадрової GT-розмітки немає. Подальший пріоритет — окрема вибірка реальних відеокадрів із негативними сценами та розділенням за вихідним роликом.
