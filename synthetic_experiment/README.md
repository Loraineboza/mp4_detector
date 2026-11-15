# Synthetic PyTorch experiment

Воспроизводимый CPU-only эксперимент по бинарной классификации синтетических данных. Внешние датасеты не скачиваются.

## Данные и задача

Генерируются две частично пересекающиеся двумерные Gaussian distributions:

- 600 объектов, по 300 на класс;
- 2 входных признака;
- стратифицированное разбиение: train — 420, validation — 90, test — 90;
- seed генератора данных: 42.

Данные создаются средствами NumPy. Признаки не нормализуются, поэтому отдельная оценка статистик нормализации не требуется.

## Архитектура

```text
Linear(2, 16)
ReLU
Linear(16, 1)
```

Выход последнего слоя — logits. Для обучения используется `BCEWithLogitsLoss`, который объединяет sigmoid и binary cross-entropy.

## Обучение и проверка

Скрипт реализует:

- фиксированные seed для Python, NumPy и PyTorch;
- явный цикл: `zero_grad`, forward, loss, `backward`, `optimizer.step`;
- validation в `model.eval()` и `torch.no_grad()`;
- выбор checkpoint по минимальному validation loss;
- однократную оценку выбранной модели на test;
- сохранение model state, optimizer state, эпохи и validation-метрик;
- загрузку checkpoint и сравнение предсказаний на фиксированном batch;
- восстановление optimizer state и продолжение обучения;
- сравнение SGD с learning rate 0.001 и 0.1 при одинаковых начальных весах, данных, порядке batch и числе эпох.

Скрипт включает ограничения детерминизма PyTorch. Совпадение результатов между разными версиями библиотек, устройствами и платформами не гарантируется.

## Установка

Требуется Python 3.10+ и PyTorch. Установите PyTorch подходящей для вашей ОС версии, следуя официальной инструкции: https://pytorch.org/get-started/locally/

Для запуска этого эксперимента torchvision и VOC 2007 не требуются.

## Запуск

Из корня репозитория:

```bash
python synthetic_experiment/experiment.py
```

Можно указать каталог артефактов:

```bash
python synthetic_experiment/experiment.py --output-dir synthetic_experiment/artifacts
```

По умолчанию артефакты сохраняются в `synthetic_experiment/artifacts/`:

- `best_checkpoint.pt` — лучший checkpoint;
- `training_log.json` — train/validation метрики по эпохам;
- `summary.json` — test-результаты, проверка восстановления и сравнение learning rate.

## Интерпретация

Test используется только один раз после выбора модели по validation. Сравнение learning rate также проводится по validation, а не по test. Поэтому test-метрики не участвуют в выборе модели или гиперпараметров.

Значения метрик формируются скриптом при запуске и сохраняются в `summary.json`. CI workflow сохраняет созданные артефакты как downloadable artifact.

## Ограничения

Эксперимент проверяет базовый PyTorch workflow на синтетической бинарной классификации. Он не измеряет качество Faster R-CNN, не является экспериментом на Pascal VOC 2007 и не заменяет полноценную оценку детектора.
