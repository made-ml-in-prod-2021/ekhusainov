"""Predicting the result from a ready-made model"""
from joblib import load

import pandas as pd

from src.enities.all_train_params import TrainingPipelineParams
from src.fit_predict.predict import preprocess_x_raw_test

LOCAL_OUTPUT = "predicts.csv"
LOCAL_PATH_CONFIG = "models/config.joblib"


def batch_predict(parametrs: TrainingPipelineParams,
                  local_output: str = LOCAL_OUTPUT,
                  ):
    """
    Load models and predict.
    """
    model = load(parametrs.output_model_path)

    # one_hot = load(parametrs.path_to_one_hot_encoder)
    # scale = load(parametrs.path_to_scaler)
    x_raw_test = pd.read_csv(parametrs.x_test_filepath)

    x_test = preprocess_x_raw_test(x_raw_test, parametrs)
    y_pred = model.predict(x_test)

    pd.DataFrame(y_pred).to_csv(local_output, index=False)


def batch_predict_command(local_path_config: str = LOCAL_PATH_CONFIG):
    """Our main function."""
    parametrs = load(local_path_config)
    batch_predict(parametrs)


if __name__ == "__main__":
    batch_predict_command()
