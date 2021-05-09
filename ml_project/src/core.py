"Our main module."
from argparse import (
    ArgumentParser,
    ArgumentDefaultsHelpFormatter,
)
from textwrap import dedent
import os.path

import pandas as pd
import numpy as np

from src.enities.all_train_params import (
    read_training_pipeline_params,
)
from src.features.build_features import build_features
from src.fit_predict.fit_model import fit_model
from src.fit_predict.predict import (
    main_predict,
    preprocess_x_raw_test,
    predict_data,
    PATH_TO_ONE_HOT_ENCODER,
    PATH_TO_SCALER,
)


DEFAULT_CONFIG_NAME = "logregr"
DEFAULT_CONFIG_PATH = "configs/logregr.yml"
DEFAULT_DATASET_FOR_PREDICT = "data/validate_part/x_test.csv"
DEFAULT_PREDICTED_DATA = "data/y_pred/y_pred.csv"


def callback_fit_predict(arguments):
    current_config_path = arguments.config_name
    current_config_path = "configs/" + current_config_path + ".yml"
    parametrs = read_training_pipeline_params(current_config_path)
    build_features(parametrs)
    fit_model(parametrs)
    main_predict(parametrs)


def callback_predict(argumets):
    parametrs = read_training_pipeline_params(DEFAULT_CONFIG_PATH)
    dataset_path = argumets.dataset
    y_pred_path = argumets.output
    model_path = parametrs.output_model_path
    if not os.path.isfile(PATH_TO_ONE_HOT_ENCODER) or \
            not os.path.isfile(PATH_TO_SCALER) or \
            not os.path.isfile(model_path):
        print(dedent("There are no models .joblib. Please run fit_predict"))
        return
    x_raw_test = pd.read_csv(dataset_path)
    x_test = preprocess_x_raw_test(
        x_raw_test,
        parametrs,
        one_hot_filepath=PATH_TO_ONE_HOT_ENCODER,
        scale_filepath=PATH_TO_SCALER,
    )
    y_pred = predict_data(x_test, parametrs)
    y_pred = pd.DataFrame(y_pred)
    y_pred.to_csv(y_pred_path, index=False)


def setup_parser(parser):
    "Argparser."
    subparsers = parser.add_subparsers(
        help="choose command",
    )

    fit_predict_parser = subparsers.add_parser(
        "fit_predict",
        help="full fit and predict",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    fit_predict_parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_NAME,
        dest="config_name",
        help="the name of the file located in configs/, without .yml, \
            default name is \"%(default)s\"",
    )
    fit_predict_parser.set_defaults(callback=callback_fit_predict)

    predict_parser = subparsers.add_parser(
        "predict",
        help="predict and save results",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    predict_parser.add_argument(
        "-d", "--dataset", default=DEFAULT_DATASET_FOR_PREDICT,
        help="the path to the dataset to predict",
    )
    predict_parser.add_argument(
        "-o", "--output", default=DEFAULT_PREDICTED_DATA,
        help="the path to predictions",
    )
    predict_parser.set_defaults(callback=callback_predict)
    # parser.add_argument(
    #     "-c", "--config",
    #     help="the name of the file located in configs/, without .yml, \
    #         default name is \"%(default)s\"",
    #     dest="config_name",
    #     default=DEFAULT_CONFIG_NAME,
    # )


# def process_arguments(config_name: str):
#     "Parse config."
#     current_config_path = config_name
#     current_config_path = "configs/" + current_config_path + ".yml"
#     parametrs = read_training_pipeline_params(current_config_path)
#     build_features(parametrs)
#     fit_model(parametrs)
#     main_predict(parametrs)


def main():
    "Our int main."
    parser = ArgumentParser(
        prog="core",
        description="Train and fit application.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)
    # process_arguments(arguments.config_name)


if __name__ == "__main__":
    main()
