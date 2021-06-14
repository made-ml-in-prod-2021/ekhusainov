import os

import click
import pandas as pd
from sklearn.metrics import accuracy_score
from joblib import load

PATH_VALIDATE_DATA = "validate.csv"
PATH_MODEL = "model.joblib"
PATH_METRIC = "metric.txt"


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
    accuracy_text = str(round(accuracy_value, 2))
    accuracy_text = "accuracy_score: " + accuracy_text + "%"
    os.makedirs(metric_path, exist_ok=True)
    with open(os.path.join(metric_path, filepath_metric), "w") as file_output:
        file_output.write(accuracy_text)


if __name__ == "__main__":
    validate()
