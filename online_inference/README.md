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
│   ├── app_params.yml
│   │                  <- Конфиг параметров rest-сервиса (ip, порт и т.п.)
│   │
│   │
│   └── core_logging.conf.yml
│                      <- Конфиг логов.
│
├── data
│   └── validate_part  <- Валидационная часть данных.
│
│
├── logs
│   └── core.log       <- Логи программы.
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

Данный вариант запуска используется для ОС Windows 10.  

Установка и создание окружения в conda:
```
conda create -n $environment_name python=3.7
conda activate $environment_name
```

Установка пакета:
```
pip install -e .
```

Запуск всех тестов из online_inference:
```
pytest -v -p no:warnings tests
```

Запуск rest-сервиса:
```
python src\fastapi_app.py
```

Отправка запросов на запущенный rest-сервис.  
В примере будет 2 запроса:  
1) Вернёт idx\:0-45 и HTTP\:200.
2) Вернёт HTTP\:400
```
python src\make_request.py
```

Сборка докер-контейнера:
```
docker build -t xluttiy/ml_in_prod_hw02_ekh:v1 .
```

Запуск докер-контейнера:
```
docker run -p 8000:8000 xluttiy/ml_in_prod_hw02_ekh:v1
```

Решить подобные ошибки:  
"Bind for 0.0.0.0:8000 failed: port is already allocated."  
Может помочь перезапуск docker-а.