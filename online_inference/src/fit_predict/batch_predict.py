"""Predicting the result from a ready-made model"""
from joblib import load

import pandas as pd

from src.enities.all_train_params import TrainingPipelineParams
from src.fit_predict.predict import preprocess_x_raw_test

LOCAL_OUTPUT = "predicts.csv"
LOCAL_PATH_CONFIG = "models/config.joblib"


def batch_predict(x_raw_test: pd.DataFrame,
                  parametrs: TrainingPipelineParams,
                  local_output: str = LOCAL_OUTPUT,
                  ):
    """
    Load models and predict.
    """
    model = load(parametrs.output_model_path)
    x_test = preprocess_x_raw_test(x_raw_test, parametrs)
    y_pred = model.predict(x_test)
    y_pred = pd.DataFrame(y_pred)
    y_pred.to_csv(local_output, index=False)

    return y_pred


def batch_predict_command(x_raw_test: pd.DataFrame,
                          local_path_config: str = LOCAL_PATH_CONFIG):
    """Our main function."""
    parametrs = load(local_path_config)
    return batch_predict(x_raw_test, parametrs)
