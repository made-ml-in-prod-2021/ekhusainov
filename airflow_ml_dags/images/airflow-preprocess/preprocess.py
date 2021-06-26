import os

import click
import pandas as pd


PATH_PREPROCESSED_DATA = "full_compare_data.csv"
PATH_TO_RAW_DATA_X = "data.csv"
PATH_TO_RAW_TARGET = "target.csv"


@click.command("preprocess")
@click.option("--raw_data_path")
@click.option("--preprocessed_data_path")
def preprocess(raw_data_path,
               preprocessed_data_path,
               filepath_x_data=PATH_TO_RAW_DATA_X,
               filepath_target=PATH_TO_RAW_TARGET,
               filepath_fulldata=PATH_PREPROCESSED_DATA):
    data_x = pd.read_csv(os.path.join(
        raw_data_path, filepath_x_data), sep=",", names=[0, 1], header=None)
    y_target = pd.read_csv(os.path.join(raw_data_path, filepath_target),
                           names=["target"], header=None)
    assert data_x.shape[0] == y_target.shape[0], (
        f"Data_shape: {data_x.shape[0]}\ny_shape: {y_target.shape[0]}"
    )
    data_x["target"] = y_target
    os.makedirs(preprocessed_data_path, exist_ok=True)
    data_x.to_csv(os.path.join(preprocessed_data_path,
                               filepath_fulldata), index=False)


if __name__ == "__main__":
    preprocess()
