from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from constants import (
    DEFAULT_ARGS,
    DATA_RAW_FOLDER,
    VOLUME,
    DATA_PREPROCESSED_FOLDER,
)

with DAG(
    "preprocess",
    default_args=DEFAULT_ARGS,
    start_date=days_ago(0),
    schedule_interval="@weekly",
) as dag:
    generate_data = DockerOperator(
        image="airflow-preprocess",
        command=f"--raw_data_path {DATA_RAW_FOLDER} --preprocessed_data_path {DATA_PREPROCESSED_FOLDER}",
        network_mode="bridge",
        task_id="preprocess_and_compare_data",
        do_xcom_push=False,
        volumes=[VOLUME],
    )
