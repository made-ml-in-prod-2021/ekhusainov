import logging
import logging.config
import sys

from typing import Tuple
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.enities.all_train_params import TrainingPipelineParams
from src.features.build_features import DEFAULT_LOGGING_PATH

import yaml

APPLICATION_NAME = "fit_model"
BUILD_FEATURES_LOGGING_CONFIG_FILEPATH = "configs/build_features_logging.conf.yml"
DEFAULT_X_TRAIN_PATH = "data/processed/x_train_for_fit_predict.csv"
DEFAULT_Y_TRAIN_PATH = "data/processed/y_train.csv"
DEFAULT_MODEL_PATH = "models/model.joblib"

logger = logging.getLogger(APPLICATION_NAME)


def setup_logging():
    "Logger from yaml config."
    with open(DEFAULT_LOGGING_PATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


def read_csv_file(
    filepath_x_train=DEFAULT_X_TRAIN_PATH,
    filepath_y_train=DEFAULT_Y_TRAIN_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    "Read preprocessed data."
    logger.info("Start reading the files.")
    x_train = pd.read_csv(filepath_x_train)
    logger.info("File %s was read", repr(filepath_x_train))
    y_train = pd.read_csv(filepath_y_train)
    logger.info("File %s was read", repr(filepath_y_train))
    return x_train, y_train


def fit_model(
    parametrs: TrainingPipelineParams,
):
    "Fit and save model."
    setup_logging()
    x_train, y_train = read_csv_file()
    y_train = y_train.values.ravel()

    current_model = parametrs.model_params.model_type
    logger.info("Start to fit data %s model.", repr(current_model))
    if "Logistic Regression" == current_model:
        model = LogisticRegression(
            penalty=parametrs.model_params.penalty,
            tol=parametrs.model_params.tol,
            C=parametrs.model_params.C,
            random_state=parametrs.model_params.random_state,
            max_iter=parametrs.model_params.max_iter,
        )
    elif "Random Forest Classifier":
        model = RandomForestClassifier(
            n_estimators=parametrs.model_params.n_estimators,
            criterion=parametrs.model_params.criterion,
            max_depth=parametrs.model_params.max_depth,
            min_samples_split=parametrs.model_params.min_samples_split,
            random_state=parametrs.model_params.random_state,

    )
    else:
        raise NotImplementedError()
    model.fit(x_train, y_train)
    logger.info("Finish to fit %s model.", repr(current_model))

    model_filepath = parametrs.output_model_path
    logger.info("Start to save model to %s", repr(model_filepath))
    dump(model, model_filepath)
    logger.info("Finish to save model to %s", repr(model_filepath))


def main():
    "Our int main."
    fit_model()


if __name__ == "__main__":
    main()
