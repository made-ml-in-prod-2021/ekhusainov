import os

from joblib import load
from sklearn.metrics import accuracy_score, f1_score
import click
import pandas as pd

PATH_METRIC = "metric.txt"
PATH_MODEL = "model.joblib"
PATH_VALIDATE_DATA = "validate.csv"


@click.command("validate")
@click.option("--model_path")
@click.option("--validate_path")
@click.option("--metric_path")
def validate(model_path, validate_path, metric_path,
             filepath_model=PATH_MODEL,
             filepath_validate_data=PATH_VALIDATE_DATA,
             filepath_metric=PATH_METRIC):
    model = load(os.path.join(
        model_path, filepath_model))
    validate_data = pd.read_csv(os.path.join(
        validate_path, filepath_validate_data))
    target_column = "target"
    y_pred = model.predict(validate_data.drop([target_column], axis=1))
    accuracy_value = accuracy_score(y_pred, validate_data[target_column]) * 100
    f1_value = f1_score(y_pred, validate_data[target_column]) * 100
    accuracy_text = str(round(accuracy_value, 2))
    f1_text = str(round(f1_value, 2))
    accuracy_text = "accuracy_score: " + accuracy_text + "%\n"
    f1_text = "f1_score: " + f1_text + "%"
    os.makedirs(metric_path, exist_ok=True)
    with open(os.path.join(metric_path, filepath_metric), "w") as file_output:
        file_output.write(accuracy_text)
        file_output.write(f1_text)


if __name__ == "__main__":
    validate()
