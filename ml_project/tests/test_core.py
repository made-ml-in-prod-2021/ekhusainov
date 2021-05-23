import numpy as np
import pandas as pd

from src.enities.all_train_params import read_training_pipeline_params
from src.features.build_features import build_features
from src.fit_predict.fit_model import fit_model
from src.fit_predict.predict import main_predict


DEFAULT_CONFIG_PATH_TEST = "tests/configs/random_forest.yml"
DEFAULT_FAKE_DATA_PATH = "tests/fake_data/fake_data.csv"
FAKE_DATASET_SIZE = 100


def test_generate_fake_data_almost_random():
    "Generating data from a similar distribution."
    parametrs = read_training_pipeline_params(DEFAULT_CONFIG_PATH_TEST)
    data = pd.read_csv(parametrs.input_data_path)
    data_fake = data.drop(data.index)
    cat_columns = parametrs.features_params.categorial_features
    num_columns = parametrs.features_params.numerical_features
    for column in num_columns:
        column_random = np.random.normal(
            loc=data[column].mean(),
            scale=data[column].std(),
            size=FAKE_DATASET_SIZE,
        )
        data_fake[column] = column_random
    target = parametrs.features_params.target_column
    cat_columns = cat_columns + [target]
    for column in cat_columns:
        count_cat_value = data[column].value_counts(normalize=True)
        cat_random = np.random.choice(
            count_cat_value.index,
            size=FAKE_DATASET_SIZE,
            p=count_cat_value.values,
        )
        data_fake[column] = cat_random
    data_fake.to_csv(DEFAULT_FAKE_DATA_PATH, index=False)
    parametrs.input_data_path = DEFAULT_FAKE_DATA_PATH
    build_features(parametrs, on_logger=False)
    fit_model(parametrs, on_logger=False)
    ac_score = main_predict(parametrs, on_logger=False)
    fake_data_generation_passed = bool(ac_score)
    assert fake_data_generation_passed, (
        f"Fake_ac_score = {ac_score}"
    )


def test_generate_fake_data_adding_gaussian_perturbation_to_num_feature():
    "Generating data from a similar distribution."
    parametrs = read_training_pipeline_params(DEFAULT_CONFIG_PATH_TEST)
    data = pd.read_csv(parametrs.input_data_path)
    data_fake = data
    cat_columns = parametrs.features_params.categorial_features
    num_columns = parametrs.features_params.numerical_features

    raw_number = data.shape[0]
    for column in num_columns:
        std_column = data[column].std()

        column_random = np.random.normal(
            loc=0,
            scale=std_column,
            size=raw_number,
        )
        data_fake[column] = data_fake[column] + column_random

    data_fake.to_csv(DEFAULT_FAKE_DATA_PATH, index=False)
    parametrs.input_data_path = DEFAULT_FAKE_DATA_PATH
    build_features(parametrs, on_logger=False)
    fit_model(parametrs, on_logger=False)
    ac_score = main_predict(parametrs, on_logger=False)
    assert ac_score > 0.5, (
        f"Fake_ac_score = {ac_score}"
    )
