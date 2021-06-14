from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

from constants import (
    DEFAULT_ARGS,
    DATA_RAW_PATH,
    VOLUME,
    DATA_PREPROCESSED_PATH,
    TRAIN_PATH,
    VALIDATE_PATH,
    MODEL_PATH,
    METRIC_PATH,
    # INTERVAL_FOR_WAIT_IN_SEC,
    # RETRY_COUNT,
)


with DAG(
    "preprocess_and_train",
    default_args=DEFAULT_ARGS,
    start_date=days_ago(0),
    schedule_interval="@weekly",
) as dag:
    # waiting_for_raw_data_x = FileSensor(
    #     task_id="waiting_for_raw_data_x",
    #     filepath=f"{DATA_RAW_PATH}/data.csv",
    #     poke_interval=10,
    #     retries=10,
    #     timeout=600,
    # )

    # waitint_for_raw_target = FileSensor(
    #     task_id="waitint_for_raw_target",
    #     filepath=f"{DATA_RAW_PATH}/target.csv",
    #     poke_interval=10,
    #     retries=10,
    #     timeout=600,
    # )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--raw_data_path {DATA_RAW_PATH} --preprocessed_data_path {DATA_PREPROCESSED_PATH}",
        network_mode="bridge",
        task_id="preprocess_and_compare_data",
        do_xcom_push=False,
        volumes=[VOLUME],
    )

    split = DockerOperator(
        image="airflow-split",
        command=f"--preprocessed_data_path {DATA_PREPROCESSED_PATH} --train_data_path {TRAIN_PATH} --validate_data_path {VALIDATE_PATH}",
        network_mode="bridge",
        task_id="split_data",
        do_xcom_push=False,
        volumes=[VOLUME],
    )

    train = DockerOperator(
        image="airflow-train",
        command=f"--train_data_path {TRAIN_PATH} --model_path {MODEL_PATH}",
        network_mode="bridge",
        task_id="train_data",
        do_xcom_push=False,
        volumes=[VOLUME],
    )

    validate = DockerOperator(
        image="airflow-validate",
        command=f"--model_path {MODEL_PATH} --validate_path {VALIDATE_PATH} --metric_path {METRIC_PATH}",
        network_mode="bridge",
        task_id="validate_data",
        do_xcom_push=False,
        volumes=[VOLUME],
    )

    # [waiting_for_raw_data_x,
    #     waitint_for_raw_target] >>
    preprocess >> split >> train >> validate
