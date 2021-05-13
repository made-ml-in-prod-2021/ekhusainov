# Структура проекта
```

├── LICENSE            <- Лицензия MIT.
│
├── README.md          <- Этот README.
│
├── Dockerfile         <- Файл для сборки докер-образа.
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
│   └── core_logging.conf.yml
│                      <- Конфиг логов.
│
├── data
│   └── validate_part  <- Валидационная часть данных..
│
│
├── logs
│   └── core.log       <- Логи программы..
│
│
├── models             <- Обученные, готовые модели в .joblib формате.
│
│
├── src                <- Исходный код проекта.
│   │
│   ├── core.py        <- Главный .py файл, его и надо запускать. Чуть ниже опишу как.
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
│   ├── fit_predict    <- Fit/predict обработанных данных.
│   │   │
│   │   ├── fit_model.py
│   │   └── predict.py
│   │
│   └── visualization  <- Генерация отчёта. 
│       │                 Для данной задачи его вполне хватает.
│       └── creating_report.py
│       
│  
└── tests              <- Тесты.

```