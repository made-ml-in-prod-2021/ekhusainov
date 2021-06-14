from datetime import timedelta


DEFAULT_ARGS = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

DATA_PREPROCESSED_PATH = "/data/processed/{{ ds }}"
DATA_RAW_PATH = "/data/raw/{{ ds }}"
METRIC_PATH = "/data/metrics/{{ ds }}"
MODEL_PATH = "/data/models/{{ ds }}"
PREDICT_PATH = "/data/predict/{{ ds }}"
TRAIN_PATH = "/data/train/{{ ds }}"
VALIDATE_PATH = "/data/validate/{{ ds }}"
VOLUME = "C:\\Users\\eh\\Documents\\GitHub\\ml_in_prod\\ekhusainov\\airflow_ml_dags\\data:/data"
