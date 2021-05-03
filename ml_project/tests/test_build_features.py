import os

import pytest
import numpy as np
import pandas as pd
from math import floor

from src.features.build_features import read_csv_file, split_to_train_test,\
    split_dataset_to_num_cat_features, categorial_feature_to_one_hot_encoding,\
    numeric_standart_scaler, concat_normalized_and_one_hot_data

RAW_DATASET_PATH = "data/raw/heart.csv"
TEST_PATH_TO_ONE_HOT_ENCODER = "models/_one_hot_test.joblib"
TEST_PATH_TO_SCALER = "models/_scaler_test.joblib"
TEST_PATH_PROCESSED_DATA = "data/processed/_heart_processed_test.csv"


@pytest.fixture()
def raw_dataset() -> pd.DataFrame:
    return pd.read_csv(RAW_DATASET_PATH)


def test_correct_size_raw_data(raw_dataset):
    current_size = raw_dataset.shape
    etalon_size = (303, 14)
    assert etalon_size == current_size, (
        f"wrong size: {current_size}"
    )


def test_correct_columns(raw_dataset):
    current_columns = raw_dataset.columns.tolist()
    etalon_columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
    ]
    assert etalon_columns == current_columns, (
        f"wrong columns: {current_columns}"
    )


@pytest.mark.parametrize(
    "test_size, etalon_answer",
    [
        pytest.param(0.15, 257),
        pytest.param(0.2, 242),
        pytest.param(0.5, 151),
    ]
)
def test_split_to_train_test(raw_dataset, test_size, etalon_answer):

    x_train, x_test, y_train, y_test = split_to_train_test(
        raw_dataset, test_size)
    train_size = x_train.shape
    test_size = x_test.shape
    etalon_train_size = (etalon_answer, 13)
    test_etalon_answer = raw_dataset.shape[0] - etalon_answer
    etalon_test_size = (test_etalon_answer, 13)
    y_train_size = len(y_train)
    y_test_size = len(y_test)
    etalon_y_train_size = etalon_answer
    etalon_y_test_size = test_etalon_answer
    assert etalon_train_size == train_size and test_size == etalon_test_size and\
        y_train_size == etalon_y_train_size and\
        y_test_size == etalon_y_test_size, (
            f"train_size: {train_size} \ntest_size: {test_size}"
        )


def test_correct_split_cat_num_features(raw_dataset):
    categorial_data, numeric_data = split_dataset_to_num_cat_features(
        raw_dataset)
    cat_columns = categorial_data.columns.tolist()
    num_columns = numeric_data.columns.tolist()
    etalon_cat_columns = [
        "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
    ]
    etalon_num_columns = [
        "age", "trestbps", "chol", "thalach", "oldpeak",
    ]
    assert etalon_cat_columns == cat_columns and\
        etalon_num_columns == num_columns, (
            f"Cat_columns: {cat_columns}\nNum_columns: {num_columns}"
        )


def test_categorial_feature_to_one_hot_encoding(raw_dataset):
    categorial_data, _ = split_dataset_to_num_cat_features(
        raw_dataset)
    one_hot_data = categorial_feature_to_one_hot_encoding(
        categorial_data, TEST_PATH_TO_ONE_HOT_ENCODER)
    num_unique = 0
    for column in categorial_data.columns.tolist():
        current_number = len(pd.unique(categorial_data[column]))
        num_unique += current_number
    assert one_hot_data.shape[0] == categorial_data.shape[0] and \
        one_hot_data.shape[1] == num_unique, (
            f"({one_hot_data.shape[0]}, {one_hot_data.shape[1]})"
    )


def test_numeric_standart_scaler(raw_dataset):
    _, numeric_data = split_dataset_to_num_cat_features(
        raw_dataset)
    normalized_data = numeric_standart_scaler(
        numeric_data, TEST_PATH_TO_SCALER)
    assert normalized_data.shape == numeric_data.shape, (
        f"{normalized_data.shape}"
    )


def test_concat_normalized_and_one_hot_data(raw_dataset):
    categorial_data, numeric_data = split_dataset_to_num_cat_features(
        raw_dataset)
    one_hot_data = categorial_feature_to_one_hot_encoding(
        categorial_data, TEST_PATH_TO_ONE_HOT_ENCODER)
    normalized_data = numeric_standart_scaler(
        numeric_data, TEST_PATH_TO_SCALER)
    finish_preprocessed_data = concat_normalized_and_one_hot_data(
        normalized_data,
        one_hot_data,
        TEST_PATH_PROCESSED_DATA,
    )
    assert finish_preprocessed_data.shape[0] == one_hot_data.shape[0] and\
        finish_preprocessed_data.shape[1] == one_hot_data.shape[1] + \
        normalized_data.shape[1], (
            f"{finish_preprocessed_data.shape}"
        )
