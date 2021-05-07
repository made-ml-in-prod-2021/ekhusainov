import os
import sys

import pytest
import numpy as np
import pandas as pd

from src.fit_predict.fit_model import (
    DEFAULT_X_TRAIN_PATH,
)
from src.fit_predict.predict import (
    read_csv_file,
    preprocess_x_raw_test,
    predict_data,
    DEFAULT_MODEL_PATH,
    DEFAULT_X_TEST_PATH,
    DEFAULT_Y_TEST_PATH,
)

from src.core import DEFAULT_CONFIG_PATH
from src.enities.all_train_params import (
    read_training_pipeline_params,
    TrainingPipelineParams,
)

def test_correct_shape_x_test_and_y_pred():
    parametrs = read_training_pipeline_params(DEFAULT_CONFIG_PATH)
    x_train = pd.read_csv(DEFAULT_X_TRAIN_PATH)
    x_raw_test, y_test = read_csv_file()
    y_test = y_test.values.ravel()
    x_test = preprocess_x_raw_test(x_raw_test, parametrs)
    y_pred = predict_data(x_test)
    assert 30 == x_test.shape[1] and y_pred.shape == y_test.shape, (
        f"Bad x_test shape: {x_test.shape}\n or bad y_pred shape {y_pred.shape}"
    )
