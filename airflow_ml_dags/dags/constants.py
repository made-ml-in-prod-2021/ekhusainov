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

# test_data = Variable.get("DEFAULT_MOUNT_FOLDER")

DATA_RAW_FOLDER = "/data/raw/{{ ds }}"
DATA_PREPROCESSED_FOLDER = "/data/processed/{{ ds }}"
VOLUME = "C:\\Users\\eh\\Documents\\GitHub\\ml_in_prod\\ekhusainov\\airflow_ml_dags\\data:/data"
# DEFAULT_MOUNT_FOLDER = [f"{test_data}:/data"]
