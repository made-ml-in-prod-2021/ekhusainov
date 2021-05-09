# Структура проекта
```

├── LICENSE            <- Лицензия MIT.
│
├── setup.py           <- Позволяет поставить проект через `pip install -e .`
│
├── requirements.txt   <- Требуемые библиотеки.
│                         Можно их поставить так: `pip install -r requirements.txt`
│                         Но они ставятся через setup.py (см. выше).
│
├── configs            <- Конфигурационные файлы.
│   │
│   ├── report_logging.conf.yml
│   │                  <- Логи для report (генерация отчёта по исходному датасету).
│   │
│   ├── core_logging.conf.yml
│   │                  <- Конфиг логов, основной программы.
│   │
│   ├── logregr.yml    <- Конфиг для логистической регрессии (по умолчанию).
│   │
│   └── random_forest.yml
│                      <- Конфиг для случайного леса.
│
├── data
│   ├── raw            <- Исходные, неменяемые данные
│   ├── └── heart.csv
│   │
│   ├── processed      <- Обработанные данные, которые для fit/predict
│   ├── validate_part  <- Валидационная часть данных.
│   └── y_pred         <- Предсказанные метки для части задания predict.
│
├── logs
│   ├── core.log       <- Логи основной программы.
│   └── report.log     <- Логи создания отчёта.
│
│
├── models             <- Обученные, готовые модели в .joblib формате.
│
│
├── report             <- Сгенерированный через ProfileReport отчёт по датасету. 
│   │                     Для данной задачи его вполне хватает.
│   ├── creating_report.py
│   └── profile_report.html
│
├── src                <- Исходный код проекта.
│   │
│   ├── enities        <- Набор "сущностей" в виде dataclass.
│   │   │
│   │   ├── all_train_params.py
│   │   │              <- Объединение всех параметров, что ниже в один dataclass.
│   │   │
│   │   ├── feature_params.py
│   │   │              <- Параметры датасета. Категориальные, числовые колонки и target.
│   │   │
│   │   ├── model_params.py
│   │   │              <- Параметры моделей. Логрегрессия и случайный лес.
│   │   │
│   │   └── train_test_split_parametrs.py
│   │                  <- Параметры train_test_split.
│   │
│   │
│   ├── features       <- Обработка сырых данных и преобразование в датасет для fit/predict.
│   │   └── build_features.py
│   │
│   └── fit_predict    <- Fit/predict обработанных данных.
│       │
│       ├── fit_model.py
│       └── predict.py
│  
└── tests              <- Тесты.

```

TO DO

запуск всех тестов из корня:  
pytest -v tests  
то же самое, только ещё посмотреть на покрытие тестами:  
pytest -v --cov --cov-branch tests  
coverage xml  
