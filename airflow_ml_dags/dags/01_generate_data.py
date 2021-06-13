from datetime import timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

from constants import (
    DEFAULT_ARGS,
    # DEFAULT_MOUNT_FOLDER,
    DATA_RAW_FOLDER,
    VOLUME,
)

with DAG(
    "generate_data",
    default_args=DEFAULT_ARGS,
    start_date=days_ago(1),
    schedule_interval="@daily",
) as dag:
    generate_data = DockerOperator(
        image="airflow-generate-data",
        command=DATA_RAW_FOLDER,
        network_mode="bridge",
        task_id="docker-airflow-generate_data",
        do_xcom_push=False,
        volumes=[VOLUME],
    )
