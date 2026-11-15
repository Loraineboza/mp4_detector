# MP4 Detector

Проект обнаружения объектов на изображениях и видео с использованием Faster R-CNN (ResNet-50 + FPN) из torchvision и датасета PASCAL VOC 2007.

В репозитории также находится независимый NumPy-модуль метрик для ограничивающих рамок: IoU, pairwise IoU, greedy NMS, class-aware NMS, Precision/Recall и VOC-style AP/mAP.

## История проекта

Первая зафиксированная запись истории Git — 27 октября 2026 года (initial commit). Это дата первого коммита репозитория, а не обязательно точная дата начала разработки.

## Состав проекта

- train.py — обучение Faster R-CNN, валидация по loss, early stopping, конфигурация и checkpoint.
- voc2007dataset.py и script.py — загрузка VOC 2007 и разбор XML-аннотаций.
- img_eval.py — инференс на изображении.
- mp4_eval.py — покадровый инференс на видео.
- detection_metrics.py — независимые NumPy-метрики и NMS.
- tests_metrics/ — тесты NumPy-метрик.
- voc_classes.py — список 20 классов VOC плюс background.
- VOCdevkit/VOC2007/ — локальные данные VOC 2007, если они присутствуют в рабочей копии.

## Архитектура

Используется torchvision.models.detection.fasterrcnn_resnet50_fpn с предобученными весами FasterRCNN_ResNet50_FPN_Weights.DEFAULT. Предиктор заменяется под 21 категорию: background и 20 классов VOC.

При первом запуске обучения torchvision может загрузить предобученные веса backbone и detector из официального хранилища PyTorch. Для инференса используются сохранённые веса проекта; модель создаётся без скачивания предобученных весов.

## Окружение

Рекомендуется Python 3.10 или новее. Установите PyTorch и torchvision как совместимую пару для вашей ОС и CUDA/CPU, используя официальный селектор установки PyTorch: https://pytorch.org/get-started/locally/

Остальные зависимости:

~~~bash
python -m pip install numpy pillow opencv-python tqdm pytest
~~~

Проверьте версии:

~~~bash
python -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__)"
~~~

## Подготовка VOC 2007

Ожидаемая структура относительно корня репозитория:

~~~text
VOCdevkit/VOC2007/
  Annotations/
  ImageSets/Main/
    train.txt
    val.txt
  JPEGImages/
~~~

Загрузчик использует train.txt и val.txt. В текущем конвейере VOC-обучения отдельная test-выборка не создаётся и итоговая test-метрика детектора не заявляется.

## Обучение

Запуск из корня репозитория:

~~~bash
python train.py
~~~

Параметры можно переопределить:

~~~bash
python train.py --epochs 30 --batch-size 2 --num-workers 2 --seed 42
~~~

Основные настройки по умолчанию:
- seed: 42;
- batch size: 2;
- максимум эпох: 30;
- optimizer: SGD, learning rate 1e-3, momentum 0.9, weight decay 1e-4;
- scheduler: StepLR, step_size=5, gamma=0.1;
- early stopping: 10 эпох без улучшения, min_delta=1e-4;
- gradient clipping: max norm 10.

Seed задаётся для Python, NumPy и PyTorch. DataLoader получает фиксированный generator и worker_init_fn. Для полной детерминированности между разными GPU, версиями библиотек и драйверами гарантия не даётся.

## Валидация

Валидационный цикл не строит граф градиентов (torch.no_grad). Для torchvision Faster R-CNN loss dictionary возвращается в training mode; в eval mode модель возвращает detections, а не loss dictionary. Поэтому для loss-based validation модель временно находится в train mode, а BatchNorm переводится в eval, чтобы не обновлять running statistics. Это не эквивалентно evaluation-mode inference и не является оценкой mAP.

## Конфигурация, веса и checkpoints

В каталоге model/ создаются:

- training_config.json — параметры запуска, версии torch/torchvision, устройство и имя enum предобученных весов;
- weights_faster_rcnn_voc.pth — state_dict лучшей модели, совместимый с текущими img_eval.py и mp4_eval.py;
- checkpoint_best.pt — полный checkpoint лучшей эпохи;
- checkpoint_last.pt — последний полный checkpoint для продолжения обучения;
- training_log.csv — train loss, validation loss и learning rate по эпохам.

Полный checkpoint содержит model_state_dict, optimizer_state_dict, scheduler_state_dict, номер эпохи, лучшую validation loss, счётчик early stopping, конфигурацию и состояния генераторов случайных чисел.

Продолжение обучения:

~~~bash
python train.py --resume
~~~

Путь к последнему checkpoint: model/checkpoint_last.pt. Параметр --epochs задаёт целевое общее число эпох, а не число дополнительных эпох.

## Инференс

Изображение:

~~~bash
python img_eval.py img/cars.png
~~~

Видео:

~~~bash
python mp4_eval.py videos/rase.mp4
~~~

При необходимости вторым аргументом передайте путь к файлу weights_faster_rcnn_voc.pth. Порог confidence по умолчанию 0.5 находится в img_eval.py и mp4_eval.py. Скрипты используют OpenCV для отображения и сохранения результатов.

## Синтетический PyTorch-эксперимент

Отдельный CPU-only эксперимент для проверки требований к обучению PyTorch без скачивания внешних датасетов:

- две частично пересекающиеся группы двумерных точек;
- train/validation/test split;
- явный цикл обучения и validation в `eval()` / `no_grad()`;
- выбор checkpoint только по validation loss;
- однократная оценка выбранной модели на test;
- сохранение и восстановление model/optimizer state;
- проверка идентичности предсказаний после загрузки и продолжение обучения;
- сравнение двух learning rate на одинаковой начальной модели и данных.

Запуск из корня репозитория:

~~~bash
python synthetic_experiment/experiment.py
~~~

Артефакты сохраняются в `synthetic_experiment/artifacts/`: checkpoint, журнал обучения и итоговый JSON. GitHub Actions также публикует их в workflow artifact с именем `pytorch-synthetic-experiment`. Подробности — в [synthetic_experiment/README.md](synthetic_experiment/README.md).

## NumPy IoU и NMS

Формат рамок: xyxy = [x_min, y_min, x_max, y_max]. Координаты трактуются как непрерывные, без добавления единицы к ширине или высоте:

area = max(0, x_max - x_min) * max(0, y_max - y_min)

Перевёрнутые или вырожденные рамки не нормализуются перестановкой координат. Если x_max <= x_min или y_max <= y_min, соответствующая ширина или высота становится нулевой через max(0, delta), площадь такой рамки равна нулю, а IoU возвращается равным нулю при нулевой площади объединения. Это соответствует фактической реализации detection_metrics.py.

pairwise_iou(boxes_a, boxes_b) возвращает матрицу формы (N, M). NMS — жадный алгоритм: кандидаты обрабатываются по убыванию score; подавление выполняется только при IoU строго больше порога. Равные scores разрешаются по исходному индексу. class_aware_nms выполняет подавление независимо для каждого класса.

Модуль метрик автономен: train.py не использует его для вычисления validation mAP. Наличие NumPy-метрик и их тестов не доказывает качество обученной нейросети.

Пример:

~~~python
from detection_metrics import pairwise_iou, nms, class_aware_nms, evaluate_map50

ious = pairwise_iou(boxes_a, boxes_b)
keep = nms(boxes, scores, iou_threshold=0.5)
keep_by_class = class_aware_nms(boxes, scores, labels, iou_threshold=0.5)
metrics = evaluate_map50(predictions, targets, class_ids=range(1, 21))
print(metrics["map"])
~~~

## Проверки

Совместимость API модели проверена отдельным CPU smoke test в GitHub Actions: PyTorch 2.15.0+cpu и torchvision 0.29.2+cpu; создание FasterRCNN с 21 классом прошло успешно. Это подтверждает совместимость указанной версии API, но не является полным обучением VOC-модели. CI run: https://github.com/Loraineboza/mp4_detector/actions/runs/43744160446.

Тесты NumPy-модуля:

~~~bash
python -m pytest -q tests_metrics
~~~

Тесты включают проверки IoU, pairwise IoU, NMS, метрик обнаружения, касания границ и вырожденных рамок. GitHub Actions проверяет NumPy-модуль и компиляцию Python-скриптов. Тесты проверяют геометрию IoU, матрицу попарных IoU, NMS и расчёт detection metrics на синтетических примерах. Они не запускают обучение Faster R-CNN.

Для воспроизводимого CPU-эксперимента без внешнего датасета см. отдельный каталог synthetic_experiment, добавленный в рамках второго этапа проверки.

## Фактические ограничения

- Полное обучение Faster R-CNN зависит от наличия VOC 2007 и вычислительных ресурсов; в рамках второго этапа не требуется запускать полное обучение.
- Текущий train.py валидирует loss, а не detection mAP.
- NumPy-метрики — отдельный модуль и не интегрированы в цикл обучения.
- Для inference требуется checkpoint, обученный с тем же числом и порядком классов.
- Датасетный загрузчик и XML-парсер являются отдельными компонентами проекта; в рамках данного исправления их логика не переписывалась.

## История изменений и вклад

До уточняющего задания в проекте были Faster R-CNN для VOC 2007, dataset loader, скрипты инференса, NumPy IoU/NMS и тесты метрик. В рамках второго этапа исправлено имя enum предобученных весов, добавлены seed и DataLoader generator, сохранение training_config.json, полные checkpoints optimizer/scheduler/RNG, восстановление обучения и журналирование.

В исходной версии train.py использовалось несуществующее имя FasterRCNN_ResNet50_FPN_Wei; оно заменено на FasterRCNN_ResNet50_FPN_Weights.DEFAULT. Также README ранее описывал воспроизводимость и checkpoints, которых не было в проверенном training script.

Исходные компоненты проекта принадлежат автору репозитория. При подготовке исправлений и документации второго этапа использовался ChatGPT как инструмент разработки и ревью. Готовая архитектура Faster R-CNN предоставляется torchvision; собственная часть проекта включает адаптацию головы под VOC, загрузчик данных, обучение/инференсные скрипты и отдельные NumPy-метрики. Полное обучение модели после этих изменений не заявляется как выполненное.
