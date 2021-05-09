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
│   ├── logregr.yml    <- Конфиг для логистической регрессии.
│   │
│   └── random_forest.yml
│                      <- Конфиг для случайного леса (по умолчанию).
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
│   └── fit_predict    <- Fit/predict обработанных данных.
│       │
│       ├── fit_model.py
│       └── predict.py
│  
└── tests              <- Тесты.

```

Установка и создание окружения в conda:
```
conda create -n $environment_name python=3.6
conda activate $environment_name
pip install -e .
```
Запуск всех тестов из ml_project:
```
pytest -v -p no:warnings tests
```
Доступен --help для запусков.
```
python src\core.py --help
python src\core.py fit_predict --help
python src\core.py predict --help
```
Для обучения пайлайна надо запускать:
```
python src\core.py fit_predict
==
python src\core.py fit_predict -c random_forest (по умолчанию)

python src\core.py fit_predict -c logregr
```
Где для -c, --config надо передать названия конфигурационного файла в configs без ".yml".

Для predict надо иметь валидационный датасет
(он будет автоматически сгенерирован при запуске предыдущей команды fit_predict).
Есть 2 параметра:

--dataset, -d
"data/validate_part/x_test.csv" (по умолчанию)
путь к датасету

--output, -o
"data/y_pred/y_pred.csv" (по умолчани)
путь к генерируемому файлу с предсказаниями 
