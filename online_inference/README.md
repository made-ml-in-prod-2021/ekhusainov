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

Скачать контейнер из хаба:
```
docker pull xluttiy/ml_in_prod_hw02_ekh:v1
```

:heavy_plus_sign: 0) ветку назовите homework2, положите код в папку online_inference
:zero:  
:heavy_plus_sign: 1) Оберните inference вашей модели в rest сервис(вы можете использовать как FastAPI, так и flask, другие желательно не использовать, дабы не плодить излишнего разнообразия для проверяющих), должен быть endpoint /predict (3 балла)
:three:  
:heavy_plus_sign: 2) Напишите тест для /predict  (3 балла) (https://fastapi.tiangolo.com/tutorial/testing/, https://flask.palletsprojects.com/en/1.1.x/testing/)
:six:  
:heavy_plus_sign: 3) Напишите скрипт, который будет делать запросы к вашему сервису -- 2 балла
:eight:  
:heavy_plus_sign: 4) Сделайте валидацию входных данных (например, порядок колонок не совпадает с трейном, типы не те и пр, в рамках вашей фантазии)  (вы можете сохранить вместе с моделью доп информацию, о структуре входных данных, если это нужно) -- 3 доп балла
https://fastapi.tiangolo.com/tutorial/handling-errors/ -- возращайте 400, в случае, если валидация не пройдена  
*В src\make_request.py во 2-м запросе сильно меньше фич, чем надо для предсказания, поэтому возвращается 400 ошибка.*
:one::one:  
:heavy_plus_sign: 5) Напишите dockerfile, соберите на его основе образ и запустите локально контейнер(docker build, docker run), внутри контейнера должен запускать сервис, написанный в предущем пункте, закоммитьте его, напишите в readme корректную команду сборки (4 балл)  
:one::five:  
:heavy_minus_sign: 6) Оптимизируйте размер docker image (3 доп балла) (опишите в readme.md что вы предприняли для сокращения размера и каких результатов удалось добиться)  -- https://docs.docker.com/develop/develop-images/dockerfile_best-practices/  
:one::five::penguin:  
:heavy_plus_sign: 7) опубликуйте образ в https://hub.docker.com/, используя docker push (вам потребуется зарегистрироваться) (2 балла)  
:one::seven:  
:heavy_plus_sign: 8) напишите в readme корректные команды docker pull/run, которые должны привести к тому, что локально поднимется на inference ваша модель (1 балл)  
Убедитесь, что вы можете протыкать его скриптом из пункта 3  
:one::eight:  
:heavy_plus_sign: 5) проведите самооценку -- 1 доп балл  
:two::zero:  
:heavy_plus_sign: 6) создайте пулл-реквест и поставьте label -- hw2  
:two::zero:

Получается :two::zero: баллов, если не ошибаюсь. :dromedary_camel: