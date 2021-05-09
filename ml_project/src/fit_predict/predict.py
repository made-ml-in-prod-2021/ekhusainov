"Predict by using .joblib file."
import logging
import logging.config
from typing import Tuple

from joblib import load
from sklearn.metrics import accuracy_score
import pandas as pd

import yaml

from src.enities.all_train_params import TrainingPipelineParams
from src.features.build_features import (
    split_dataset_to_cat_num_features,
    concat_normalized_and_one_hot_data,
    DEFAULT_LOGGING_PATH,
)

APPLICATION_NAME = "predict_model"
DEFAULT_X_TEST_PATH = "data/validate_part/x_test.csv"
DEFAULT_Y_TEST_PATH = "data/validate_part/y_test.csv"
DEFAULT_MODEL_PATH = "models/model.joblib"
PATH_TO_ONE_HOT_ENCODER = "models/one_hot.joblib"
PATH_TO_SCALER = "models/standart_scaler.joblib"

logger = logging.getLogger(APPLICATION_NAME)


def setup_logging():
    "Logger from yaml config."
    with open(DEFAULT_LOGGING_PATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


def read_csv_file(filepath_x_test=DEFAULT_X_TEST_PATH,
                  filepath_y_test=DEFAULT_Y_TEST_PATH,
                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    "Read validate data."
    logger.info("Start reading the files.")
    x_raw_test = pd.read_csv(filepath_x_test)
    logger.info("File %s was read", repr(filepath_x_test))
    y_test = pd.read_csv(filepath_y_test)
    logger.info("File %s was read", repr(filepath_y_test))
    return x_raw_test, y_test


def preprocess_x_raw_test(x_raw_test: pd.DataFrame,
                          parametrs: TrainingPipelineParams,
                          one_hot_filepath=PATH_TO_ONE_HOT_ENCODER,
                          scale_filepath=PATH_TO_SCALER,
                          on_logger=False,
                          ) -> pd.DataFrame():
    "Use .joblib objects for preprocess."
    if on_logger:
        setup_logging()
    logger.info("Split test data to num and categorial.")
    categorial_data, numeric_data = split_dataset_to_cat_num_features(
        x_raw_test, parametrs)
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


def predict_data(x_test: pd.DataFrame,
                 parametrs: TrainingPipelineParams,
                 ) -> pd.DataFrame:
    "Predict data by .joblib."
    logger.info("Start predict data.")
    model_filepath = parametrs.output_model_path
    model = load(model_filepath)
    y_pred = model.predict(x_test)
    logger.info("Finish predict data.")
    return y_pred


def main_predict(parametrs: TrainingPipelineParams,
                 on_logger=True):
    "Our main function in this module."
    if on_logger:
        setup_logging()
    x_raw_test, y_test = read_csv_file()
    x_test = preprocess_x_raw_test(x_raw_test, parametrs)
    y_pred = predict_data(x_test, parametrs)
    ac_score = accuracy_score(y_pred, y_test)
    return ac_score
