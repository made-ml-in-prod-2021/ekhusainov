# Структура проекта

├── LICENSE            <- Лицензия MIT.
│
├── README.md          <- Документация по проекту.
│
├── setup.py           <- Позволяет поставить проект через `pip install -e`
│
├── requirements.txt   <- Требуемые библиотеки.
│                         Можно их поставить так: `pip install -r requirements.txt`
│
├── configs            <- Конфигурационные файлы.
│   └── report_logging.conf.yml
│
├── data
│   └── raw            <- Исходные, неменяемые данные
│       └── heart.csv
│
├── models             <- Обученные, готовые модели (пока нет).
│
├── notebooks          <- Jupyter notebooks. Пока нет, да походу и не будет.
│
├── report             <- Сгенерированный через ProfileReport отчёт по датасету.  
│   ├── creating_report.py
│   └── profile_report.html
│
│
├── src                <- Исходный код проекта.
│   ├── __init__.py    <- Делает src Python модулем.
│   │
│   ├── data           <- Пока нет.
│   │   └── make_dataset.py
│   │
│   ├── features       <- Пока нет.
│   │   └── build_features.py
│   │
│   └── models         <- Пока нет.
│       │                 
│       ├── predict_model.py
│       └── train_model.py
│   
│
└── tests              <- Тесты.
    ├── __init__.py
    └── test_report.py



# 1) Структура проекта

## 1.1) ~/

Установка всех нужных библиотек:
pip install -r requirements.txt

## 1.2) data
### 1.2.1) raw

Наш исходный датасет. Я не стал его помещать в отдельное хранилище, только из-за того, что он весит жалкие 12-13Кб.

## 1.3) notebooks

Скорее всего ограничусь папкой report.

## 1.4) report

Автоматические сгенерированный отчёт, с помощью библиотеки  
from pandas_profiling import ProfileReport

## 1.5) configs

Конфиги yaml.

### 1.5.1) report_logging.conf.yml

Конфиг логгирования отчёта.

## 1.6) test

запуск всех тестов из корня:
pytest -v tests
то же самое, только ещё посмотреть на покрытие тестами:
pytest -v --cov --cov-branch tests
coverage xml
