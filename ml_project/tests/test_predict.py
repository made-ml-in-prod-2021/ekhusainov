from src.fit_predict.predict import (
    read_csv_file,
    preprocess_x_raw_test,
    predict_data,
)

from tests.test_core import DEFAULT_CONFIG_PATH_TEST
from src.enities.all_train_params import read_training_pipeline_params


def test_correct_shape_x_test_and_y_pred():
    parametrs = read_training_pipeline_params(DEFAULT_CONFIG_PATH_TEST)
    x_raw_test, y_test = read_csv_file(
        parametrs
    )
    y_test = y_test.values.ravel()
    x_test = preprocess_x_raw_test(x_raw_test, parametrs)
    y_pred = predict_data(x_test, parametrs)
    assert x_test.shape[1] >= 1 and y_pred.shape == y_test.shape, (
        f"Bad x_test shape: {x_test.shape}\n or bad y_pred shape {y_pred.shape}"
    )
