# Структура проекта
```

├── LICENSE            <- Лицензия MIT.  
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
│   ├── creating_report.  
│   └── profile_report.html   
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
  
```

TO DO

запуск всех тестов из корня:
pytest -v tests
то же самое, только ещё посмотреть на покрытие тестами:
pytest -v --cov --cov-branch tests
coverage xml
