# Розширений датасет і запуск

**5 377 train / 600 validation / 499 test**, один клас bird із 200 видів CUB. Дані й кеш не включені в Git або ZIP.

[Джерело CUB](https://www.vision.caltech.edu/datasets/cub_200_2011/) · [Зафіксована копія](https://huggingface.co/datasets/bentrevett/caltech-ucsd-birds-200-2011/tree/1ef09e021b0b65b40337f6f285909656f407f6e0/data). Використання — некомерційні дослідження та навчання.

## Аудит

Чотири файли джерела перевірені за SHA256. Відхилено 30 некоректних bbox у повному джерелі; попередні validation/test виключені з train. Додатково відсіяно кадр, схожий на старий test. Групи дублікатів зведені до одного представника.

Повторний аудит: **0 помилок, 0 знайдених пар дублікатів, 0 перетинів вибірок**. dHash — евристика, не доказ відсутності всіх схожих кадрів. [Стислий JSON](expanded_dataset.json); повні протоколи — у `datasets/bird-expanded/`.

## Час

Benchmark 128 train / 64 validation при 640, CPU: **55 с**. Прогноз 3 епох — 63 хв; фактично повний запуск із фінальними оцінками зайняв **48 хв**. Ваги вже включені до релізу. [Результати](improvement.md).

## Команди Warp

З кореня проєкту після встановлення залежностей:

```bash
source .venv/bin/activate
python prepare_data.py       # Пропустити, якщо datasets/bird уже існує.
python prepare_expanded.py   # Пропустити, якщо розширений набір уже існує.
python benchmark_expanded.py # Необов’язковий повторний замір; нове --name для повтору.
```

Для нового донавчання поточних ваг незалежно від термінала:

```bash
mkdir -p work
nohup caffeinate -i .venv/bin/python -u train_boxes.py \
  --data datasets/bird-expanded/data.yaml --model weights/best.pt \
  --imgsz 640 --epochs 3 --lr0 0.00001 --name expanded-next \
  > work/expanded-next.log 2>&1 < /dev/null &
echo $! > work/expanded-next.pid
```

Це новий експеримент, не повторне відтворення релізу. Повна послідовність від початкової YOLO26n — у [README](../README.md).

`nohup` дозволяє закрити термінал, `caffeinate` блокує автоматичний сон. Кришку ноутбука залишити відкритою. Ваги зберігаються після епох; автоматичного відновлення bbox-етапу немає. Журнал — `work/expanded-next.log`, результат — `runs/experiments/expanded-next/weights/best.pt`.

Кандидата порівнювати з чинною моделлю на тих самих 600 validation-зображеннях при 640. Поточний test уже оцінений — не використовувати його для подальшого пошуку параметрів.
