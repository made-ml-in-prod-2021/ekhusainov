"""Preparing data for training."""
from textwrap import dedent
import logging
import logging.config
import sys

from typing import Tuple
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import pandas as pd
import click

# TODO походу убрать и запускать из core
from train_test_split_parametrs import TrainTestSplitParametrs


import yaml


APPLICATION_NAME = "build_features"
BUILD_FEATURES_LOGGING_CONFIG_FILEPATH = "../../configs/build_features_logging.conf.yml"
PATH_TO_DATASET = "../../data/raw/heart.csv"
PATH_TO_ONE_HOT_ENCODER = "../../models/one_hot.joblib"
PATH_TO_SCALER = "../../models/standart_scaler.joblib"
PREPROCESSED_DATA_FILEPATH = "../../data/processed/x_train_for_fit_predict.csv"
X_TEST_FILEPATH = "../../data/validate_part/x_test.csv"
Y_TEST_FILEPATH = "../../data/validate_part/y_test.csv"
Y_TRAIN_FILEPATH = "../../data/processed/y_train.csv"

logger = logging.getLogger(APPLICATION_NAME)


def setup_logging():
    "Logger from yaml config."
    with open(BUILD_FEATURES_LOGGING_CONFIG_FILEPATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


def read_csv_file(filepath: str) -> pd.DataFrame:
    "Read raw data."
    logger.debug("Start reading the file.")
    data = pd.read_csv(filepath)
    logger.info("File %s was read", repr(filepath))
    return data


def split_to_train_test(
    data: pd.DataFrame,
    parametrs: TrainTestSplitParametrs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    "Split raw data to x_train, x_test, y_train, y_test."
    logger.debug("Start to split datatest to train and test.")
    x_data = data.drop(['target'], axis=1)
    target = data['target'].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, target, test_size=parametrs.test_size,
        random_state=parametrs.random_state,
        stratify=target,
    )
    logger.info("Finish split datatest to train and test.")
    return x_train, x_test, y_train, y_test


def split_dataset_to_cat_num_features(x_data: pd.DataFrame) -> tuple:
    "One data split to tuple (categorial_data, num_data)."
    logger.debug("Start to split dataset to numeric and categorial features")
    columns_x_data = x_data.columns.tolist()
    if "target" in columns_x_data:
        logger.info("The full dataset with \"target\" is given for input")
        x_data = x_data.drop(['target'], axis=1)
    cat_columns = [
        "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
    ]
    num_columns = [
        "age", "trestbps", "chol", "thalach", "oldpeak",
    ]
    categorial_data = x_data[cat_columns]
    numeric_data = x_data[num_columns]
    logger.info(dedent("""\
        Finished dividing the dataset into categorical and \
        numeric variables.
    """))
    return categorial_data, numeric_data


def categorial_feature_to_one_hot_encoding(
        categorial_data: pd.DataFrame,
        filepath: str, ) -> np.array:
    "Transform categorial features to one hot encoding and safe model"
    logger.debug("Start to one hot encoding.")
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore")
    one_hot_encoder.fit(categorial_data)
    transformed_to_one_hot = one_hot_encoder.transform(
        categorial_data).toarray()
    logger.info("Finish one hot encoding.")
    save_data_transformer(one_hot_encoder, filepath)
    return transformed_to_one_hot


def numeric_standard_scaler(
        numeric_data: pd.DataFrame,
        filepath: str, ) -> np.array:
    "Normalize numeric data and save scaler model."
    logger.debug("Begin scale numeric data.")
    scaler = StandardScaler()
    scaler.fit(numeric_data)
    normalized_data = scaler.transform(numeric_data)
    logger.info("Finish scale numeric data")
    save_data_transformer(scaler, filepath)
    return normalized_data


def concat_normalized_and_one_hot_data(
        normalized_data: np.array,
        one_hot_data: np.array,) -> pd.DataFrame:
    "Concat two dataframe to fit/predict version and save one hot model."
    logger.debug("Start concatenate norm and one hot data.")
    normalized_data = pd.DataFrame(normalized_data)
    one_hot_data = pd.DataFrame(one_hot_data)
    preprocessed_data = pd.concat([normalized_data, one_hot_data], axis=1)
    logger.info("Finish concatenate norm and one hot data.")
    return preprocessed_data


def save_file_to_csv(dataset: pd.DataFrame, filepath: str):
    "Saving dataset by filepath in csv format."
    logger.debug("Start saving dataset to %s.", repr(filepath))
    dataset.to_csv(filepath, index=False)
    logger.info("Finish saving dataset to %s.", repr(filepath))


def save_data_transformer(transformer: object, filepath: str):
    "Saving data transformer by filepath in joblib format."
    logger.debug("Start saving transformer to %s.", repr(filepath))
    dump(transformer, filepath)
    logger.info("Finish saving transformer to %s.", repr(filepath))


@click.command(name="build_features")
def build_features():
    raw_data = read_csv_file(PATH_TO_DATASET)
    x_train, x_test, y_train, y_test = split_to_train_test(raw_data, parametrs)
    save_file_to_csv(x_test, X_TEST_FILEPATH)
    save_file_to_csv(pd.DataFrame(y_train), Y_TRAIN_FILEPATH)
    save_file_to_csv(pd.DataFrame(y_test), Y_TEST_FILEPATH)
    categorial_data, numeric_data = split_dataset_to_cat_num_features(x_train)
    one_hot_data = categorial_feature_to_one_hot_encoding(
        categorial_data, PATH_TO_ONE_HOT_ENCODER)
    normilized_data = numeric_standard_scaler(numeric_data, PATH_TO_SCALER)
    finish_preprocessed_data = concat_normalized_and_one_hot_data(
        normilized_data, one_hot_data)
    save_file_to_csv(finish_preprocessed_data, PREPROCESSED_DATA_FILEPATH)


def main():
    "Our int main."
    setup_logging()
    build_features()


if __name__ == "__main__":
    main()
