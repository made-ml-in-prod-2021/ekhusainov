import os
import sys

import pytest
import numpy as np
import pandas as pd

from src.fit_predict.fit_model import (
    DEFAULT_X_TRAIN_PATH,
)
from src.fit_predict.predict import (
    preprocess_x_raw_test,
    DEFAULT_X_TEST_PATH,
)


def test_correct_shape_x_test():
    x_train = pd.read_csv(DEFAULT_X_TRAIN_PATH)
    x_raw_test = pd.read_csv(DEFAULT_X_TEST_PATH)
    x_test = preprocess_x_raw_test(x_raw_test)
    assert x_train.shape[1] == x_test.shape[1], (
        f"Bad x_test shape: {x_test.shape}"
    )
