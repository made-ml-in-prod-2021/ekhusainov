from joblib import load
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    read_csv_file, split_to_train_test,
    split_dataset_to_cat_num_features,
    categorial_feature_to_one_hot_encoding,
    numeric_standard_scaler, concat_normalized_and_one_hot_data,
    save_file_to_csv, save_data_transformer,
)
from src.enities.all_train_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from src.core import DEFAULT_CONFIG_PATH

RAW_DATASET_PATH = "data/raw/heart.csv"
# TEST_PATH_PROCESSED_DATA = "data/processed/_heart_processed_test.csv"
# TEST_PATH_TO_ONE_HOT_ENCODER = "models/_one_hot_test.joblib"
# TEST_PATH_TO_SCALER = "models/_scaler_test.joblib"


@pytest.fixture()
def raw_dataset() -> pd.DataFrame:
    return pd.read_csv(RAW_DATASET_PATH)


@pytest.fixture()
def parametrs() -> TrainingPipelineParams:
    return read_training_pipeline_params(DEFAULT_CONFIG_PATH)


@pytest.fixture()
def temp_filepath(tmp_path):
    directory = tmp_path / "sub"
    directory.mkdir()
    filepath = "temp.joblib"
    temp_filepath = directory / filepath
    return temp_filepath


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
def test_split_to_train_test(raw_dataset, test_size, etalon_answer, parametrs):
    parametrs.splitting_params.test_size = test_size
    x_train, x_test, y_train, y_test = split_to_train_test(
        raw_dataset, parametrs)
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
            f"""train_size: {train_size} \ntest_size: {test_size}
            param.test_size: {parametrs.splitting_params.test_size}"""
        )


def test_correct_split_cat_num_features(raw_dataset, parametrs):
    categorial_data, numeric_data = split_dataset_to_cat_num_features(
        raw_dataset, parametrs)
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


def test_categorial_feature_to_one_hot_encoding(raw_dataset, temp_filepath, parametrs):
    categorial_data, _ = split_dataset_to_cat_num_features(
        raw_dataset, parametrs)
    one_hot_data = categorial_feature_to_one_hot_encoding(
        categorial_data, temp_filepath)
    num_unique = 0
    for column in categorial_data.columns.tolist():
        current_number = len(pd.unique(categorial_data[column]))
        num_unique += current_number
    assert one_hot_data.shape[0] == categorial_data.shape[0] and \
        one_hot_data.shape[1] == num_unique, (
            f"({one_hot_data.shape[0]}, {one_hot_data.shape[1]})")


def test_numeric_standard_scaler(raw_dataset, temp_filepath, parametrs):
    _, numeric_data = split_dataset_to_cat_num_features(raw_dataset, parametrs)
    normalized_data = numeric_standard_scaler(numeric_data, temp_filepath)
    assert normalized_data.shape == numeric_data.shape, (
        f"{normalized_data.shape}"
    )


def test_concat_normalized_and_one_hot_data(raw_dataset, temp_filepath, parametrs):
    categorial_data, numeric_data = split_dataset_to_cat_num_features(
        raw_dataset, parametrs)
    one_hot_data = categorial_feature_to_one_hot_encoding(
        categorial_data, temp_filepath)
    normalized_data = numeric_standard_scaler(numeric_data, temp_filepath)
    finish_preprocessed_data = concat_normalized_and_one_hot_data(
        normalized_data,
        one_hot_data,
    )
    assert finish_preprocessed_data.shape[0] == one_hot_data.shape[0] and\
        finish_preprocessed_data.shape[1] == one_hot_data.shape[1] + \
        normalized_data.shape[1], (
            f"{finish_preprocessed_data.shape}")


def test_save_file_to_csv(raw_dataset, tmp_path):
    directory = tmp_path / "sub"
    directory.mkdir()
    filepath = "temp.csv"
    temp_filepath = directory / filepath
    save_file_to_csv(raw_dataset, temp_filepath)
    assert bool(temp_filepath.read_text()) and\
        pd.read_csv(temp_filepath).shape == (303, 14), ("Fail save.")


def test_save_data_transformer(tmp_path):
    directory = tmp_path / "sub"
    directory.mkdir()
    filepath = "temp.joblib"
    temp_filepath = directory / filepath
    one_hot = OneHotEncoder(handle_unknown='ignore')
    one_hot.fit(pd.DataFrame([1, 2]))
    save_data_transformer(one_hot, temp_filepath)
    model_from_file = load(temp_filepath)
    etalon_output = np.array([[1., 0.], [0., 1.], [0., 0.]])
    current_output = model_from_file.transform(pd.DataFrame([1, 2, 3])).toarray()
    assert np.array_equal(etalon_output, current_output), ("Fail save.")


def test_logging(capsys, caplog):
    with caplog.at_level("DEBUG"):
        raw_data = read_csv_file(RAW_DATASET_PATH)
        captured = capsys.readouterr()
        
        assert captured.out == ""
        assert captured.err == ""
        assert any("Start" in message for message in caplog.messages)
