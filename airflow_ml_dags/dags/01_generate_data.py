from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from constants import (
    DEFAULT_ARGS,
    DATA_RAW_PATH,
    VOLUME,
)

with DAG(
    "01_generate_data",
    default_args=DEFAULT_ARGS,
    start_date=days_ago(0),
    schedule_interval="@daily",
) as dag:
    generate_data = DockerOperator(
        image="airflow-generate-data",
        command=DATA_RAW_PATH,
        network_mode="bridge",
        task_id="docker-airflow-generate_data",
        do_xcom_push=False,
        volumes=[VOLUME],
    )
