import os

from sklearn.model_selection import train_test_split
import click
import pandas as pd

PATH_PREPROCESSED_DATA = "full_compare_data.csv"
PATH_TRAIN_DATA = "train.csv"
PATH_VALIDATE_DATE = "validate.csv"


@click.command("split")
@click.option("--preprocessed_data_path")
@click.option("--train_data_path")
@click.option("--validate_data_path")
def split(preprocessed_data_path,
          train_data_path,
          validate_data_path,
          filepath_fulldata=PATH_PREPROCESSED_DATA,
          filepath_train_data=PATH_TRAIN_DATA,
          filepath_validate_data=PATH_VALIDATE_DATE):
    full_data = pd.read_csv(os.path.join(
        preprocessed_data_path, filepath_fulldata), sep=",")
    train_data, validate_data = train_test_split(
        full_data, test_size=0.3, stratify=full_data["target"],
        shuffle=True, random_state=1)
    os.makedirs(train_data_path, exist_ok=True)
    train_data.to_csv(os.path.join(train_data_path,
                                   filepath_train_data), index=False)
    os.makedirs(validate_data_path, exist_ok=True)
    validate_data.to_csv(os.path.join(validate_data_path,
                                      filepath_validate_data), index=False)


if __name__ == "__main__":
    split()
