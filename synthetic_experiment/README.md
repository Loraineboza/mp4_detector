# Synthetic PyTorch experiment

Небольшой воспроизводимый эксперимент для проверки базового PyTorch training workflow. Работает на CPU, использует только синтетические данные и не скачивает датасеты.

## Состав проверки

- фиксированный seed Python, NumPy и PyTorch;
- раздельные train/validation/test выборки (420/90/90);
- небольшая MLP-регрессия и явный цикл обучения;
- validation выполняется с model.eval() и torch.no_grad();
- лучший checkpoint выбирается по validation MSE;
- checkpoint содержит модель, optimizer, эпоху и лучшую метрику;
- проверяется побитовая идентичность предсказаний после загрузки;
- optimizer state восстанавливается и выполняется дополнительная эпоха обучения;
- сравниваются две скорости обучения при одинаковых данных, начальных весах, batch order и количестве эпох.

## Установка

Используйте Python 3.10+ и совместимую пару PyTorch/torchvision. Для этого эксперимента torchvision не требуется.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
~~~

## Запуск

Из каталога synthetic_experiment:

~~~bash
python experiment.py
~~~

Артефакты сохраняются в artifacts/: best_checkpoint.pt, training_log.json и summary.json.

## Фактический запуск

Основной воспроизводимый CI-прогон выполнен на CPU с PyTorch 2.15.0+cpu, seed=42. Отдельный локальный запуск в рабочей среде также завершился успешно на PyTorch 2.10.0+cpu; небольшие отличия последних знаков метрик ожидаемы между версиями PyTorch.

- best epoch: 28
- best validation MSE: 0.0738621261384752
- test MSE: 0.2303723282284207
- predictions after checkpoint reload: exactly equal
- restored optimizer state entries: 4
- training continued after optimizer restoration: yes
- continued training MSE: 0.11135560336212318

Сравнение learning rate после 20 эпох при одинаковых данных и начальном состоянии:
- SGD lr=0.001: validation MSE 2.256805335150825
- SGD lr=0.1: validation MSE 0.13025523291693794

Подробный журнал по эпохам находится в training_log.json, итоговая машинно-читаемая сводка — в summary.json. Успешный CI run: https://github.com/Loraineboza/mp4_detector/actions/runs/43743825163.

## Ограничения

Это синтетическая регрессионная задача для проверки цикла обучения, валидации, checkpoint/resume и влияния learning rate. Она не измеряет качество Faster R-CNN и не заменяет эксперимент на реальном датасете.
