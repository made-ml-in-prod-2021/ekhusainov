from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from constants import (
    DEFAULT_ARGS,
    DATA_RAW_PATH,
    MODEL_PATH,
    PREDICT_PATH,
    VOLUME,
)

with DAG(
    "predict_data",
    default_args=DEFAULT_ARGS,
    start_date=days_ago(0),
    schedule_interval="@daily",
) as dag:
    generate_data = DockerOperator(
        image="airflow-predict",
        command=f"--data_x_path {DATA_RAW_PATH} --model_path {MODEL_PATH} --predict_path {PREDICT_PATH}",
        network_mode="bridge",
        task_id="predict",
        do_xcom_push=False,
        volumes=[VOLUME],
    )
