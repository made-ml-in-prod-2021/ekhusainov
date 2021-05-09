from src.fit_predict.fit_model import (
    DEFAULT_X_TRAIN_PATH,
    DEFAULT_Y_TRAIN_PATH,
    read_csv_file,
)


def test_read_data():
    x_train, y_train = read_csv_file(
        DEFAULT_X_TRAIN_PATH, DEFAULT_Y_TRAIN_PATH)
    assert y_train.shape[1] == 1 and x_train.shape[1] > 5, (
        f"{x_train.shape[1]} and {y_train.shape[1]}"
    )
