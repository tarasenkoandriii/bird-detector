# Сторіборд презентації Bird Detector

Коміт джерела експериментів: `f64dd6601c1f5ab6cf87e1444672711a95a0d2e6`. Усього слайдів: 60.

## Слайд 01. Bird Detector YOLO26n
- **На слайді:** Українська презентація про навчання, перевірку та відеодетекцію птахів.
- **Візуалізація:** `hero`
- **Джерело:** README.md; examples/output.mp4

## Слайд 02. Завдання та результат
- **На слайді:** Відео → покадрові рамки → клас, confidence і готовий MP4.
- **Візуалізація:** `flow`
- **Джерело:** README.md; predict_video.py

## Слайд 03. Карта дослідження
- **На слайді:** Незмінна pretrained модель → A–D → окрема v2 серія.
- **Візуалізація:** `lineage`
- **Джерело:** README.md; reports/model_profiles.json; reports/video_study/README.md

## Слайд 04. Визначення baseline
- **На слайді:** Pretrained COCO YOLO26n; варіант A уже адаптований.
- **Візуалізація:** `baseline`
- **Джерело:** reports/pretrained_vs_profiles.md; reports/training_provenance.json

## Слайд 05. Обчислювальний стенд
- **На слайді:** CPU і MPS вимірювалися окремо на одному MacBook Pro.
- **Візуалізація:** `specs`
- **Джерело:** reports/stand_benchmark/README.md

## Слайд 06. Доказова база
- **На слайді:** README, первинні JSON/CSV, протоколи, приклади та хеші.
- **Візуалізація:** `sources`
- **Джерело:** README.md; reports/; examples/*.metrics.json

## Слайд 07. CUB: фото для детекції
- **На слайді:** Один клас bird; розподіл 5377 / 600 / 499.
- **Візуалізація:** `counts`
- **Джерело:** README.md; THIRD_PARTY.md

## Слайд 08. Legacy відеодані
- **На слайді:** 87 навчальних прикладів і 32 validation з двох інших послідовностей.
- **Візуалізація:** `counts`
- **Джерело:** reports/video_training.md; reports/video_data_audit.json

## Слайд 09. Підготовка розмітки
- **На слайді:** Нормалізація й аудит зменшують ризик витоку; не прибирають його математично.
- **Візуалізація:** `audit`
- **Джерело:** reports/video_data_audit.json; reports/project_audit/verification.md

## Слайд 10. Походження A–D
- **На слайді:** B, C і D — окремі гілки від фотоорієнтира A.
- **Візуалізація:** `lineage`
- **Джерело:** reports/training_provenance.json; reports/model_profiles.json

## Слайд 11. Умови A–D
- **На слайді:** Спільний старт не означає зміну лише одного параметра.
- **Візуалізація:** `variants`
- **Джерело:** reports/video_training_protocol.json; reports/profile_d_protocol.json

## Слайд 12. Як читати mAP і різницю
- **На слайді:** Абсолютна оцінка — %, різниця між оцінками — відсоткові пункти.
- **Візуалізація:** `metric`
- **Джерело:** reports/model_profiles.md; reports/video_study/README.md

## Слайд 13. Pretrained на CUB
- **На слайді:** Вихідна pretrained YOLO26n: 78,40% на 600 фото.
- **Візуалізація:** `bigMetric`
- **Джерело:** reports/pretrained_final_validation.json; reports/pretrained_vs_profiles.csv

## Слайд 14. CUB: абсолютні оцінки
- **На слайді:** На фото A–D вищі за вихідну модель; різниці між A–D невеликі.
- **Візуалізація:** `cubBars`
- **Джерело:** reports/pretrained_vs_profiles.csv; reports/model_profiles.json

## Слайд 15. CUB: зміна до pretrained
- **На слайді:** A +2,68 · B +2,41 · C +2,67 · D +2,77 в.п.
- **Візуалізація:** `deltaCub`
- **Джерело:** reports/pretrained_vs_profiles.csv; reports/model_profiles.json

## Слайд 16. Legacy video: абсолютні оцінки
- **На слайді:** Pretrained 56,36%; A–D нижче за цим історичним протоколом.
- **Візуалізація:** `videoBars`
- **Джерело:** reports/pretrained_vs_profiles.csv; reports/model_profiles.json

## Слайд 17. Legacy video: signed Δ
- **На слайді:** A −5,40 · B −4,42 · C −5,70 · D −4,38 в.п.
- **Візуалізація:** `deltaVideo`
- **Джерело:** reports/pretrained_vs_profiles.csv; reports/model_profiles.json

## Слайд 18. Фото та legacy video разом
- **На слайді:** П’ять точок показують компроміс двох validation метрик.
- **Візуалізація:** `scatterLegacy`
- **Джерело:** reports/pretrained_vs_profiles.csv

## Слайд 19. Окремо: історичне D проти A
- **На слайді:** D–A: +0,09 в.п. CUB та +1,02 в.п. legacy video.
- **Візуалізація:** `daCompare`
- **Джерело:** reports/model_profiles.json; reports/profile_d_backgrounds.json

## Слайд 20. Чому D — основний серед A–D
- **На слайді:** D проходить збережене правило відносно A; це локальний вибір.
- **Візуалізація:** `decision`
- **Джерело:** reports/model_profiles.json; README.md

## Слайд 21. Legacy і v2 рамки
- **На слайді:** Legacy tracking рамка могла не включати ноги; v2 охоплює видимого птаха.
- **Візуалізація:** `gt`
- **Джерело:** reports/video_study/annotation_audit.md; reports/video_study/data/README.md

## Слайд 22. Зміна GT, незмінні ваги
- **На слайді:** 51,98% → 77,97% не є приростом ваг D.
- **Візуалізація:** `gtCompare`
- **Джерело:** reports/video_study/baselines.json; reports/model_profiles.json

## Слайд 23. Обсяг v2 validation
- **На слайді:** 24 позитивні кадри й 8 фонових вирізок із двох послідовностей.
- **Візуалізація:** `counts`
- **Джерело:** reports/video_study/README.md; reports/video_study/data/manifest.json

## Слайд 24. Створення v2 розмітки
- **На слайді:** Асистент розмічав без прогнозів; незалежного людського аудиту немає.
- **Візуалізація:** `audit`
- **Джерело:** reports/video_study/annotation_audit.md; reports/video_study/data/audit.json

## Слайд 25. Навчальні приклади v2
- **На слайді:** 102 унікальні відеоприклади + повтори + CUB = 164 пред’явлення.
- **Візуалізація:** `formula`
- **Джерело:** reports/video_study/README.md; reports/video_study/data/manifest.json

## Слайд 26. Що розморожувалося
- **На слайді:** Head-only у більшості запусків; один рецепт також розморожував neck.
- **Візуалізація:** `layers`
- **Джерело:** reports/video_study/README.md; reports/video_study/runs/*/protocol.json

## Слайд 27. Спільний протокол коротких запусків
- **На слайді:** CPU · 640 px · batch 8 · AdamW · LR 3e−6 · seed 26.
- **Візуалізація:** `protocol`
- **Джерело:** reports/video_study/README.md; reports/video_study/runs/*/protocol.json

## Слайд 28. Raw N і D: різні старти
- **На слайді:** Pretrained N/head −1,2461 в.п.; D/head +0,5271 в.п.
- **Візуалізація:** `studyStart`
- **Джерело:** reports/video_study/raw_n.json; reports/video_study/candidate.json

## Слайд 29. Окремий challenger S
- **На слайді:** S зросла на 0,2457 в.п., але її старт нижчий за N.
- **Візуалізація:** `studyS`
- **Джерело:** reports/video_study/raw_s.json

## Слайд 30. П’ять коротких запусків
- **На слайді:** Один успіх не стирає невдалий raw-N/head і скромні результати інших рецептів.
- **Візуалізація:** `studyAll`
- **Джерело:** reports/video_study/comparison.json; reports/video_study/README.md

## Слайд 31. Δ до власного epoch zero
- **На слайді:** Δ рахуються від старту кожного рецепта; старт лишається кандидатом.
- **Візуалізація:** `studyDelta`
- **Джерело:** reports/video_study/comparison.json; reports/video_study/baselines.json

## Слайд 32. Head3 та свіжий head5
- **На слайді:** Head3 checkpoint: епоха 2; head5: епоха 1.
- **Візуалізація:** `headCompare`
- **Джерело:** reports/video_study/head_budget_comparison.json; reports/video_study/README.md

## Слайд 33. Head проти neck + head
- **На слайді:** 78,4981% проти 78,0153%; переваги neck у цій серії немає.
- **Візуалізація:** `neckCompare`
- **Джерело:** reports/video_study/comparison.json; reports/video_study/runs/video-v2-d-neck-3ep/protocol.json

## Слайд 34. Фото/відео компроміс
- **На слайді:** Head3: +0,5271 відео та −0,0894 CUB пункту; head5: +0,4329 і −0,0772.
- **Візуалізація:** `photoVideo`
- **Джерело:** reports/video_study/candidate.json; reports/video_study/head_budget_comparison.json; reports/video_study/photo_guard.json

## Слайд 35. Відомі точки v2
- **На слайді:** Scatter показує лише моделі з виміряними CUB і v2 координатами.
- **Візуалізація:** `scatterV2`
- **Джерело:** reports/video_study/photo_guard.json; reports/video_study/comparison.json; reports/video_study/baselines.json

## Слайд 36. Фоновий контроль v2
- **На слайді:** 0/8 фонів із FP для D, head3 та head5; вибірка мала.
- **Візуалізація:** `background`
- **Джерело:** reports/video_study/backgrounds_D0.json; backgrounds_head3.json; backgrounds_head5.json

## Слайд 37. Tiled inference: механізм
- **На слайді:** Повний кадр + tiles → повернення координат → один global NMS.
- **Візуалізація:** `tiles`
- **Джерело:** reports/video_study/tiled_recipe.md; reports/video_study/tiled_d_head3/protocol.json

## Слайд 38. F0/F1: парна якість
- **На слайді:** 74,9868% → 41,5870%; різниця −33,3998 в.п.
- **Візуалізація:** `tilesAP`
- **Джерело:** reports/video_study/tiled_d_head3/results.json; summary.json

## Слайд 39. F0/F1: швидкість і вартість
- **На слайді:** 10,79 → 1,37 кадру/с; 32 → 176 forward-викликів.
- **Візуалізація:** `tilesFPS`
- **Джерело:** reports/video_study/tiled_d_head3/results.json; summary.json

## Слайд 40. Рішення щодо tiles
- **На слайді:** Поточну конфігурацію F1 відхилено; дрібні птахи не тестувалися.
- **Візуалізація:** `decision`
- **Джерело:** reports/video_study/README.md; reports/video_study/tiled_recipe.md

## Слайд 41. Матриця виміряних напрямів
- **На слайді:** Head має скромний плюс; neck, S і tiles не показали підстав змінювати D.
- **Візуалізація:** `matrix`
- **Джерело:** reports/video_study/README.md; reports/major_improvements.md

## Слайд 42. Історичний seed-дослід D
- **На слайді:** П’ять seed із того самого старту A за незмінного протоколу.
- **Візуалізація:** `seeds`
- **Джерело:** reports/seed_study/README.md; summary.json

## Слайд 43. П’ять точок та розкид
- **На слайді:** D: CUB 81,22 ± 0,04; legacy video 51,51 ± 0,38 в.п. SD.
- **Візуалізація:** `seedResults`
- **Джерело:** reports/seed_study/summary.json

## Слайд 44. Контроль CPU проти MPS
- **На слайді:** Одна прогріта train-епоха: 24,98 с CPU і 27,36 с MPS.
- **Візуалізація:** `benchmark`
- **Джерело:** reports/stand_benchmark/README.md; results.json

## Слайд 45. Час і пропускна здатність
- **На слайді:** Показники CPU/MPS збережено окремо з однаковим малим рецептом.
- **Візуалізація:** `benchmark`
- **Джерело:** reports/stand_benchmark/results.json

## Слайд 46. RAM та MPS memory
- **На слайді:** RSS 3024/3490 MiB; MPS allocated і driver — окремі лічильники.
- **Візуалізація:** `memory`
- **Джерело:** reports/stand_benchmark/results.json

## Слайд 47. Приклад роботи на відео
- **На слайді:** 240/240 кадрів; демонстрація не має незалежної покадрової GT.
- **Візуалізація:** `videoDemo`
- **Джерело:** examples/output_video_focus.metrics.json; reports/video_study/README.md

## Слайд 48. Помилки та межі прикладів
- **На слайді:** FP/FN називаємо лише там, де є відповідна GT.
- **Візуалізація:** `errors`
- **Джерело:** reports/project_audit/README.md; reports/video_study/annotation_audit.md

## Слайд 49. Аудит проєкту
- **На слайді:** Вибір моделі, remap COCO, захист файлів і 46 перевірок.
- **Візуалізація:** `audit`
- **Джерело:** reports/project_audit/findings.md; reports/project_audit/verification.md

## Слайд 50. Відтворення вимірів
- **На слайді:** D: weights/best.pt · head3: weights/video_focus.pt.
- **Візуалізація:** `commands`
- **Джерело:** README.md; reports/video_study/README.md

## Слайд 51. Межі доказів
- **На слайді:** Мала development-validation, один seed і жодної нової незалежної відеотест-вибірки.
- **Візуалізація:** `limits`
- **Джерело:** reports/video_study/README.md; reports/video_study/data/audit.json

## Слайд 52. Карта майбутніх напрямів
- **На слайді:** Пріоритет — незалежні сцени й повна розмітка; інше має окремий go/no-go.
- **Візуалізація:** `roadmap`
- **Джерело:** reports/major_improvements.md; reports/video_study/plan.json

## Слайд 53. Нові незалежні відеосцени
- **На слайді:** Різні камери, відстані, фони, польоти та негативні ролики.
- **Візуалізація:** `plan`
- **Джерело:** reports/major_improvements.md

## Слайд 54. Повна розмітка всіх птахів
- **На слайді:** Видимі птахи, правила рамок, контроль якості та версії GT.
- **Візуалізація:** `plan`
- **Джерело:** reports/video_study/annotation_audit.md; reports/major_improvements.md

## Слайд 55. Незалежний video test
- **На слайді:** Групування за послідовністю запобігає витоку сусідніх кадрів.
- **Візуалізація:** `plan`
- **Джерело:** reports/major_improvements.md; README.md

## Слайд 56. Консервативне head донавчання
- **На слайді:** Кілька seed, фіксовані умови, фото guard і чіткий критерій.
- **Візуалізація:** `plan`
- **Джерело:** reports/video_study/README.md; reports/major_improvements.md

## Слайд 57. Коли повертатися до SAHI
- **На слайді:** Лише на окремому наборі дрібних птахів і з парним NMS.
- **Візуалізація:** `plan`
- **Джерело:** reports/video_study/tiled_recipe.md; reports/major_improvements.md

## Слайд 58. Distillation S → N
- **На слайді:** Сильний bird учитель і trainable ознаки — передумови, не доведений шлях.
- **Візуалізація:** `plan`
- **Джерело:** reports/video_study/raw_s.json; reports/major_improvements.md

## Слайд 59. P2 і часові методи
- **На слайді:** Архітектурна голова, згладжування та tracking — різні гіпотези.
- **Візуалізація:** `plan`
- **Джерело:** reports/major_improvements.md

## Слайд 60. Підсумок вимірів
- **На слайді:** Основний D незмінний; head3 — експериментальний; pretrained лишається головним baseline.
- **Візуалізація:** `summary`
- **Джерело:** README.md; reports/pretrained_vs_profiles.csv; reports/video_study/README.md
