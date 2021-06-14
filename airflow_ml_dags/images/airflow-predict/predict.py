import os

import sklearn
import click
import pandas as pd
import numpy as np
from joblib import load

PATH_TO_RAW_DATA_X = "data.csv"
PATH_MODEL = "model.joblib"
PATH_PREDICT = "predict.csv"


@click.command("predict")
@click.option("--data_x_path")
@click.option("--model_path")
@click.option("--predict_path")
def predict(data_x_path, model_path, predict_path,
            filepath_x_data=PATH_TO_RAW_DATA_X,
            filepath_model=PATH_MODEL,
            filepath_predict=PATH_PREDICT):
    data_x = pd.read_csv(os.path.join(
        data_x_path, filepath_x_data), sep=",", names=[0, 1], header=None)
    model = load(os.path.join(
        model_path, filepath_model))
    os.makedirs(predict_path, exist_ok=True)
    y_pred = model.predict(data_x)
    y_pred = np.array(y_pred)
    np.savetxt(os.path.join(predict_path, filepath_predict), y_pred, fmt="%i")


if __name__ == "__main__":
    predict()
