"""Predicting the result from a ready-made model"""
import logging
from textwrap import dedent
from typing import Tuple
from joblib import load
from fastapi import HTTPException

import numpy as np
import pandas as pd

from src.enities.all_train_params import TrainingPipelineParams
from src.enities.logging_params import setup_logging

APPLICATION_NAME = "batch_predict"
HTTP_BAD_REQUEST = 400
LOCAL_PATH_CONFIG = "models/config.joblib"

logger = logging.getLogger(APPLICATION_NAME)


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


def batch_predict(models_tuple: Tuple,
                  x_raw_test: pd.DataFrame,
                  parametrs: TrainingPipelineParams,
                  ):
    """
    Load models and predict.
    """
    feature_list = parametrs.features_params.categorial_features + \
        parametrs.features_params.numerical_features
    current_columns = x_raw_test.columns.tolist()
    if len(current_columns) < len(feature_list):
        error_msg = "The number of features is too small."
        logger.error(error_msg)
        raise HTTPException(
            detail=error_msg,
            status_code=HTTP_BAD_REQUEST,
        )
    logger.info("Start to predict data.")

    # The object looks like this:
    # models_tuple = tuple([
    #     model,
    #     one_hot_code_model,
    #     scale_model,
    # ])
    model = models_tuple[0]

    x_test = preprocess_x_raw_test(x_raw_test, parametrs)
    y_pred = model.predict(x_test)
    logger.info("Finish to predict data.")
    y_pred = pd.DataFrame(y_pred)
    return y_pred


def batch_predict_command(models_tuple: Tuple,
                          x_raw_test: pd.DataFrame,
                          local_path_config: str = LOCAL_PATH_CONFIG,
                          ):
    """Our main function."""
    setup_logging()
    parametrs = load(local_path_config)
    return batch_predict(models_tuple, x_raw_test, parametrs)
