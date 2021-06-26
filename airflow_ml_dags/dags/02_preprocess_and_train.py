from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from constants import (
    DATA_PREPROCESSED_PATH,
    DATA_RAW_PATH,
    DEFAULT_ARGS,
    METRIC_PATH,
    MODEL_PATH,
    TRAIN_PATH,
    VALIDATE_PATH,
    VOLUME,
)


with DAG(
    "02_preprocess_and_train",
    default_args=DEFAULT_ARGS,
    schedule_interval="@weekly",
    start_date=days_ago(0),
) as dag:
    preprocess = DockerOperator(
        command=f"--raw_data_path {DATA_RAW_PATH} --preprocessed_data_path {DATA_PREPROCESSED_PATH}",
        do_xcom_push=False,
        image="airflow-preprocess",
        network_mode="bridge",
        task_id="preprocess_and_compare_data",
        volumes=[VOLUME],
    )

    split = DockerOperator(
        command=f"--preprocessed_data_path {DATA_PREPROCESSED_PATH} --train_data_path {TRAIN_PATH} --validate_data_path {VALIDATE_PATH}",
        do_xcom_push=False,
        image="airflow-split",
        network_mode="bridge",
        task_id="split_data",
        volumes=[VOLUME],
    )

    train = DockerOperator(
        command=f"--train_data_path {TRAIN_PATH} --model_path {MODEL_PATH}",
        do_xcom_push=False,
        image="airflow-train",
        network_mode="bridge",
        task_id="train_data",
        volumes=[VOLUME],
    )

    validate = DockerOperator(
        command=f"--model_path {MODEL_PATH} --validate_path {VALIDATE_PATH} --metric_path {METRIC_PATH}",
        do_xcom_push=False,
        image="airflow-validate",
        network_mode="bridge",
        task_id="validate_data",
        volumes=[VOLUME],
    )

    preprocess >> split >> train >> validate
