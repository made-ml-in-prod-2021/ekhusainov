from src.fit_predict.fit_model import (
    # DEFAULT_X_TRAIN_PATH,
    # DEFAULT_Y_TRAIN_PATH,
    read_csv_file,
)
from src.core import DEFAULT_CONFIG_PATH
from src.enities.all_train_params import read_training_pipeline_params


def test_read_data():
    parametrs = read_training_pipeline_params(DEFAULT_CONFIG_PATH)
    x_train, y_train = read_csv_file(parametrs)
    assert y_train.shape[1] == 1 and x_train.shape[1] > 5, (
        f"{x_train.shape[1]} and {y_train.shape[1]} and {repr(parametrs)}"
    )
