# MIMEx

Репозиторий содержит реализацию Masked Autoencoder (MAE) для обучения представлений в задачах обучения с подкреплением. Основные компоненты:

- Реализация MAE с Vision Transformer
- Replay Buffer для работы с данными
- Интеграция с DM Control для работы с окружениями
- Логирование в Weights & Biases
- Генерация видео с реконструкциями

## Установка

```bash
pip install -r requirements.txt
```

## Использование

Для запуска обучения:

```bash
python train.py --config src/configs/config.yaml
```

## Структура проекта

```
src/
├── configs/           # Конфигурационные файлы
├── datasets/          # Реализация replay buffer и окружения
├── model/            # Реализация MAE
├── utils/            # Утилиты для логирования и видео
└── logger/           # Интеграция с Weights & Biases
```

## Лицензия

MIT
