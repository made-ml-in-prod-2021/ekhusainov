from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from constants import (
    DATA_RAW_PATH,
    DEFAULT_ARGS,
    MODEL_PATH,
    PREDICT_PATH,
    VOLUME,
)

with DAG(
    "03_predict_data",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=days_ago(0),
) as dag:
    generate_data = DockerOperator(
        command=f"--data_x_path {DATA_RAW_PATH} --model_path {MODEL_PATH} --predict_path {PREDICT_PATH}",
        do_xcom_push=False,
        image="airflow-predict",
        network_mode="bridge",
        task_id="predict",
        volumes=[VOLUME],
    )
