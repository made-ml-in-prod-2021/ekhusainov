"""Predicting the result from a ready-made model"""
import logging
from joblib import load
from fastapi import HTTPException

import pandas as pd

from src.enities.all_train_params import TrainingPipelineParams
from src.enities.logging_params import setup_logging
from src.fit_predict.predict import preprocess_x_raw_test

APPLICATION_NAME = "batch_predict"
HTTP_BAD_REQUEST = 400
LOCAL_OUTPUT = "predicts.csv"
LOCAL_PATH_CONFIG = "models/config.joblib"

logger = logging.getLogger(APPLICATION_NAME)


def batch_predict(x_raw_test: pd.DataFrame,
                  parametrs: TrainingPipelineParams,
                  local_output: str = LOCAL_OUTPUT,
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
    model = load(parametrs.output_model_path)
    x_test = preprocess_x_raw_test(x_raw_test, parametrs)
    y_pred = model.predict(x_test)
    logger.info("Finish to predict data.")
    y_pred = pd.DataFrame(y_pred)
    y_pred.to_csv(local_output, index=False)
    return y_pred


def batch_predict_command(x_raw_test: pd.DataFrame,
                          local_path_config: str = LOCAL_PATH_CONFIG):
    """Our main function."""
    setup_logging()
    parametrs = load(local_path_config)
    return batch_predict(x_raw_test, parametrs)
