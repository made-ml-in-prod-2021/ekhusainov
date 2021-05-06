import logging
import logging.config
import sys

from typing import Tuple
import numpy as np
import pandas as pd
from joblib import load

import yaml

from src.features.build_features import (
    split_dataset_to_cat_num_features,
    concat_normalized_and_one_hot_data,
)


APPLICATION_NAME = "predict_model"
BUILD_FEATURES_LOGGING_CONFIG_FILEPATH = "configs/build_features_logging.conf.yml"
DEFAULT_X_TEST_PATH = "data/validate_part/x_test.csv"
DEFAULT_Y_TEST_PATH = "data/validate_part/y_test.csv"
DEFAULT_MODEL_PATH = "models/model.joblib"
PATH_TO_ONE_HOT_ENCODER = "models/one_hot.joblib"
PATH_TO_SCALER = "models/standart_scaler.joblib"

logger = logging.getLogger(APPLICATION_NAME)


def setup_logging():
    "Logger from yaml config."
    with open(BUILD_FEATURES_LOGGING_CONFIG_FILEPATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


def read_csv_file(
    filepath_x_test=DEFAULT_X_TEST_PATH,
    filepath_y_test=DEFAULT_Y_TEST_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    "Read validate data."
    logger.info("Start reading the files.")
    x_raw_test = pd.read_csv(filepath_x_test, sep="\t")
    logger.info("File %s was read", repr(filepath_x_test))
    y_test = pd.read_csv(filepath_y_test)
    logger.info("File %s was read", repr(filepath_y_test))
    return x_raw_test, y_test


def preprocess_x_raw_test(
    x_raw_test: pd.DataFrame,
    one_hot_filepath=PATH_TO_ONE_HOT_ENCODER,
    scale_filepath=PATH_TO_SCALER,

) -> pd.DataFrame():
    logger.info("Split test data to num and categorial.")
    categorial_data, numeric_data = split_dataset_to_cat_num_features(
        x_raw_test)
    logger.info("Finish split test data.")

    logger.info("Read one hot and scale models.")
    one_hot_code_model = load(one_hot_filepath)
    scale = load(scale_filepath)
    logger.info("Finish read one hot and scale models.")

    logger.info("Start to transform test data.")
    one_hot_data = one_hot_code_model.transform(categorial_data).toarray()
    normalized_data = scale.transform(numeric_data)
    logger.info("Finish transform test data.")

    x_test = concat_normalized_and_one_hot_data(normalized_data, one_hot_data)
    return x_test


def main():
    "Our int main."
    pass


if __name__ == "__main__":
    main()
