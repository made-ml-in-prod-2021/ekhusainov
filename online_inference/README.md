# Структура проекта
```
├── LICENSE            <- Лицензия MIT.
│
│
├── README.md          <- Этот README.
│
│
├── Dockerfile         <- Файл для сборки докер-образа.
│
│
├── setup.py           <- Позволяет поставить проект через `pip install -e .`
│
│
├── requirements.txt   <- Требуемые библиотеки.
│                         Можно их поставить так: `pip install -r requirements.txt`
│                         Но они ставятся через setup.py (см. выше).
│
│
├── configs            <- Конфигурационные файлы.
│   │
│   │
│   ├── report_logging.conf.yml
│   │                  <- Логи для report (генерация отчёта по исходному датасету).
│   │
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
│   ├── fastapi_app.py
│   │                  <- Главный .py файл, который запускает rest-сервис.
│   │
│   ├── make_request.py
│   │                  <- Создание запроса и отправка через rest-сервис.
│   │
│   │
│   ├── enities        <- Набор "сущностей" в виде dataclass.
│   │
│   │
│   ├── features       <- Обработка сырых данных и преобразование в датасет для fit/predict.
│   │
│   │
│   └── fit_predict    <- Fit/predict обработанных данных и предсказание target.
│       
│  
└── tests              <- Тесты.
```

Установка приложения:

```
pip install -e .
```

Запуск rest-сервиса:

```
python src\fastapi_app.py
```

Отправка запросов на запущенный rest-сервис:

```
python src\fastapi_app.py
```

Сборка докер-контейнера:

```
docker build -t eldar/online_inference:v1 .
```

Запуск докер-контейнера:

```
docker run -p 8000:8000 eldar/online_inference:v1
```