import os

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump

PATH_TRAIN_DATA = "train.csv"
PATH_MODEL = "model.joblib"


@click.command("train")
@click.option("--train_data_path")
@click.option("--model_path")
def train(train_data_path, model_path,
          filepath_train_data=PATH_TRAIN_DATA,
          filepath_model=PATH_MODEL):
    train_data = pd.read_csv(os.path.join(
        train_data_path, filepath_train_data))
    logistic_model = LogisticRegression()
    target_column = "target"
    x_train = train_data.drop([target_column], axis=1)
    y_train = train_data[target_column]
    logistic_model.fit(x_train, y_train)
    os.makedirs(model_path, exist_ok=True)
    dump(logistic_model, os.path.join(model_path, filepath_model))


if __name__ == "__main__":
    train()
