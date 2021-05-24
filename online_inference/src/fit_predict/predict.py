"""Predict by using .joblib file."""
import logging
import logging.config
from typing import Tuple

from joblib import load
import pandas as pd

import yaml

from src.enities.all_train_params import TrainingPipelineParams
from src.features.build_features import (
    split_dataset_to_cat_num_features,
    concat_normalized_and_one_hot_data,
    DEFAULT_LOGGING_PATH,
)

APPLICATION_NAME = "predict_model"

logger = logging.getLogger(APPLICATION_NAME)


def setup_logging():
    """Logger from yaml config."""
    with open(DEFAULT_LOGGING_PATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


def preprocess_x_raw_test(x_raw_test: pd.DataFrame,
                          parametrs: TrainingPipelineParams,
                          on_logger=False,
                          models_tuple=False,
                          ) -> pd.DataFrame():
    """Use .joblib objects for preprocess."""
    if on_logger:
        setup_logging()

    logger.info("Split test data to num and categorial.")
    categorial_data, numeric_data = split_dataset_to_cat_num_features(
        x_raw_test, parametrs)
    logger.info("Finish split test data.")

    if not models_tuple:
        one_hot_filepath = parametrs.path_to_one_hot_encoder
        scale_filepath = parametrs.path_to_scaler
        logger.info("Read one hot and scale models.")
        one_hot_code_model = load(one_hot_filepath)
        scale = load(scale_filepath)
        logger.info("Finish read one hot and scale models.")
    else:
        # The object looks like this:
        # models_tuple = tuple([
        #     model,
        #     one_hot_code_model,
        #     scale_model,
        # ])
        one_hot_code_model = models_tuple[1]
        scale = models_tuple[2]
        logger.info("Model is ready to work (from input).")

    logger.info("Start to transform test data.")
    one_hot_data = one_hot_code_model.transform(categorial_data).toarray()
    normalized_data = scale.transform(numeric_data)
    logger.info("Finish transform test data.")

    x_test = concat_normalized_and_one_hot_data(normalized_data, one_hot_data)
    return x_test
