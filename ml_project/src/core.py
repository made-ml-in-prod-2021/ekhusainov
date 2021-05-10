"Our main module."
from argparse import (
    ArgumentParser,
    ArgumentDefaultsHelpFormatter,
)
import logging
import logging.config
import os.path

import pandas as pd
from joblib import load

import yaml

from src.enities.all_train_params import (
    read_training_pipeline_params,
)
from src.features.build_features import (
    build_features,
    DEFAULT_LOGGING_PATH,
)
from src.fit_predict.fit_model import (
    fit_model,
    CONFIG_FOR_CURRENT_MODEL_PATH,
)
from src.fit_predict.predict import (
    main_predict,
    preprocess_x_raw_test,
    predict_data,
)


APPLICATION_NAME = "core"
DEFAULT_CONFIG_NAME = "random_forest"
DEFAULT_CONFIG_PATH = "configs/random_forest.yml"
DEFAULT_DATASET_FOR_PREDICT = "data/validate_part/x_test.csv"
DEFAULT_PREDICTED_DATA = "data/y_pred/y_pred.csv"

logger = logging.getLogger(APPLICATION_NAME)


def setup_logging():
    "Logger from yaml config."
    with open(DEFAULT_LOGGING_PATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


def callback_fit_predict(arguments):
    "Argparse fit_predict."
    setup_logging()
    current_config_path = arguments.config_name
    current_config_path = "configs/" + current_config_path + ".yml"
    parametrs = read_training_pipeline_params(current_config_path)
    build_features(parametrs, on_logger=False)
    fit_model(parametrs, on_logger=False)
    ac_score = main_predict(parametrs, on_logger=False)
    model_name = parametrs.model_params.model_type
    logger.info("Model %s done. Accuracy score is equal to %s.",
                model_name, repr(ac_score))
    print(model_name)
    print(f"Accuracy score: {ac_score}")


def callback_predict(argumets):
    "Argparse predict only."
    setup_logging()
    if not os.path.isfile(CONFIG_FOR_CURRENT_MODEL_PATH):
        logger.error("The model has not been started before, fit_preidict it.")
        return
    parametrs = load(CONFIG_FOR_CURRENT_MODEL_PATH)
    dataset_path = argumets.dataset
    y_pred_path = argumets.output
    model_path = parametrs.output_model_path
    if not os.path.isfile(parametrs.path_to_one_hot_encoder) or \
            not os.path.isfile(parametrs.path_to_scaler) or \
            not os.path.isfile(model_path):
        logger.error("There are no models .joblib. Please run fit_predict")
        return
    x_raw_test = pd.read_csv(dataset_path)
    x_test = preprocess_x_raw_test(x_raw_test, parametrs)
    y_pred = predict_data(x_test, parametrs)
    y_pred = pd.DataFrame(y_pred)
    logger.info("Start saving predicted data in %s", repr(y_pred_path))
    y_pred.to_csv(y_pred_path, index=False)
    print(f"The file was saved at the path {repr(y_pred_path)}")
    logger.info("Finish saving predicted data in %s", repr(y_pred_path))


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


if __name__ == "__main__":
    main()
