import os
from datetime import timedelta


from airflow.models import Variable

DEFAULT_ARGS = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

DATA_RAW_FOLDER = "/data/raw/{{ ds }}"
DATA_PREPROCESSED_FOLDER = "/data/processed/{{ ds }}"
VOLUME = "C:\\Users\\eh\\Documents\\GitHub\\ml_in_prod\\ekhusainov\\airflow_ml_dags\\data:/data"
TRAIN_PATH = "/data/train/{{ ds }}"
VALIDATE_PATH = "/data/validate/{{ ds }}"