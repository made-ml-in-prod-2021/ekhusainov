import os
import sys

import pytest
import numpy as np
import pandas as pd

from src.fit_predict.fit_model import (
    read_csv_file,
    fit_model,
    DEFAULT_X_TRAIN_PATH,
    DEFAULT_Y_TRAIN_PATH,
)


def test_read_data():
    x_train, y_train = read_csv_file(
        DEFAULT_X_TRAIN_PATH, DEFAULT_Y_TRAIN_PATH)
    assert 1 == y_train.shape[1] and x_train.shape[1] > 5, (
        f"{x_train.shape[1]} and {y_train.shape[1]}"
    )
