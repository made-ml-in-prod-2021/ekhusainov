import logging
import logging.config
import sys

import numpy as np
import pandas as pd

import yaml

APPLICATION_NAME = "fit_predict"
BUILD_FEATURES_LOGGING_CONFIG_FILEPATH = "configs/build_features_logging.conf.yml"
DEFAULT_X_TRAIN_PATH = "data/processed/x_train_for_fit_predict.csv"
DEFAULT_Y_TRAIN_PATH = "data/processed/y_train.csv"


logger = logging.getLogger(APPLICATION_NAME)


def setup_logging():
    "Logger from yaml config."
    with open(BUILD_FEATURES_LOGGING_CONFIG_FILEPATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


def read_csv_file(
    filepath_x_train=DEFAULT_X_TRAIN_PATH,
    filepath_y_train=DEFAULT_Y_TRAIN_PATH,
) -> pd.DataFrame:
    "Read preprocessed data."
    logger.info("Start reading the files.")
    x_train = pd.read_csv(filepath_x_train)
    logger.info("File %s was read", repr(filepath_x_train))
    y_train = pd.read_csv(filepath_y_train)
    logger.info("File %s was read", repr(filepath_y_train))
    return x_train, y_train


def fit_model


def main():
    pass


if __name__ == "__main__":
    main()
