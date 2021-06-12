from datetime import timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

DEFAULT_ARGS = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "generate_data",
    default_args=DEFAULT_ARGS,
    start_date=days_ago(1),
    schedule_interval="@daily",
) as dag:
    generate_data = DockerOperator(
        image="airflow-generate-data",
        network_mode="bridge",
        task_id="docker-airflow-generate_data",
        # do_xcom_push=False,
        # volumes=[f"{Variable.get("data_path")}:/data"],
    )
