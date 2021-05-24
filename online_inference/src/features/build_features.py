"""Preparing data for training."""
from textwrap import dedent
from typing import Tuple
import logging
import logging.config

import numpy as np
import pandas as pd

import yaml

from src.enities.all_train_params import TrainingPipelineParams


APPLICATION_NAME = "build_features"
DEFAULT_LOGGING_PATH = "configs/core_logging.conf.yml"

logger = logging.getLogger(APPLICATION_NAME)


def setup_logging():
    """Logger from yaml config."""
    with open(DEFAULT_LOGGING_PATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


def split_dataset_to_cat_num_features(x_data: pd.DataFrame,
                                      parametrs: TrainingPipelineParams,
                                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    "One data split to tuple (categorial_data, num_data)."
    logger.info("Start to split dataset to numeric and categorial features")
    columns_x_data = x_data.columns.tolist()
    if "target" in columns_x_data:
        logger.info("The full dataset with \"target\" is given for input")
        x_data = x_data.drop(['target'], axis=1)
    cat_columns = parametrs.features_params.categorial_features
    num_columns = parametrs.features_params.numerical_features
    categorial_data = x_data[cat_columns]
    numeric_data = x_data[num_columns]
    logger.info(dedent("""\
        Finished dividing the dataset into categorical and 
        numeric variables.""").replace("\n", ""))
    return categorial_data, numeric_data


def concat_normalized_and_one_hot_data(
        normalized_data: np.array,
        one_hot_data: np.array,) -> pd.DataFrame:
    """Concat two dataframe to fit/predict version and save one hot model."""
    logger.info("Start concatenate norm and one hot data.")
    normalized_data = pd.DataFrame(normalized_data)
    one_hot_data = pd.DataFrame(one_hot_data)
    preprocessed_data = pd.concat([normalized_data, one_hot_data], axis=1)
    logger.info("Finish concatenate norm and one hot data.")
    return preprocessed_data
