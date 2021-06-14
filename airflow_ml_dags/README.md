# Структура проекта
```
├── LICENSE            <- Лицензия MIT.
│
│
├── README.md          <- Этот README.
│
│
├── docker-compose.yml <- Конфиги
│
│
├── dags               <- Все даги. (Directed acyclic graph.)
│
│
├── data               <- Данные для экспериментов и результаты.
│
│
├── images             <- Образы docker и код job-ов дагов.
│
│
└── notebook           <- Краткое визуальное представление, как выглядят наши исходные данные.

```

Данный вариант запуска используется для ОС Windows 10.

Запуск всех докер-образов:
```
docker-compose up --build
```

Убить запущенные докер-образы:
```
docker-compose down
```

Интерфейс airflow будет тут:
```
http://localhost:8080/
```

Логин: admin  
Пароль: admin


:heavy_plus_sign: 0) Поднимите airflow локально, используя docker compose (можно использовать из примера https://github.com/made-ml-in-prod-2021/airflow-examples/) :zero:
:heavy_plus_sign: 1) (5 баллов) Реализуйте dag, который генерирует данные для обучения модели (генерируйте данные, можете использовать как генератор синтетики из первой дз, так и что-то из датасетов sklearn), вам важно проэмулировать ситуации постоянно поступающих данных
- записывайте данные в /data/raw/{{ ds }}/data.csv, /data/raw/{{ ds }}/target.csv :five:

:heavy_plus_sign: 2) (10 баллов) Реализуйте dag, который обучает модель еженедельно, используя данные за текущий день. В вашем пайплайне должно быть как минимум 4 стадии, но дайте волю своей фантазии=)

- подготовить данные для обучения(например, считать из /data/raw/{{ ds }} и положить /data/processed/{{ ds }}/train_data.csv)
- расплитить их на train/val
- обучить модель на train (сохранить в /data/models/{{ ds }} 
- провалидировать модель на val (сохранить метрики к модельке) :one::five:

:heavy_plus_sign: 3) Реализуйте dag, который использует модель ежедневно (5 баллов)
- принимает на вход данные из пункта 1 (data.csv)
- считывает путь до модельки из airflow variables(идея в том, что когда нам нравится другая модель и мы хотим ее на прод 
- делает предсказание и записывает их в /data/predictions/{{ds }}/predictions.csv :two::zero:

:heavy_minus_sign: 3а)  Реализуйте сенсоры на то, что данные готовы для дагов тренировки и обучения (3 доп балла)

:heavy_plus_sign: 4) вы можете выбрать 2 пути для выполнения ДЗ. 
-- все даги реализованы только с помощью DockerOperator (10 баллов) (пример https://github.com/made-ml-in-prod-2021/airflow-examples/blob/main/dags/11_docker.py). :three::zero:

:heavy_minus_sign: 5) Протестируйте ваши даги (5 баллов) https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html :three::zero:
:heavy_minus_sign: 6) В docker compose так же настройте поднятие mlflow и запишите туда параметры обучения, метрики и артефакт(модель) (5 доп баллов) :three::zero:
:heavy_minus_sign: 7) вместо пути в airflow variables  используйте апи Mlflow Model Registry (5 доп баллов)
Даг для инференса подхватывает последнюю продакшен модель. 
:heavy_minus_sign: 8) Настройте alert в случае падения дага (3 доп. балла) :three::zero:
https://www.astronomer.io/guides/error-notifications-in-airflow
:heavy_plus_sign: 9) традиционно, самооценка (1 балл) :three::one:

