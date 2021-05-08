import os
from argparse import (
    ArgumentParser,
    ArgumentDefaultsHelpFormatter,
)
from src.features.build_features import build_features
from src.fit_predict.fit_model import fit_model
from src.fit_predict.predict import main_predict
from src.enities.all_train_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)

DEFAULT_CONFIG_PATH = "configs/logregr.yml"


def setup_parser(parser):
    parser.add_argument(
        "-c", "--config",
        help="path to config file, default path is \"%(default)s\"",
        dest="config_path",
        default=DEFAULT_CONFIG_PATH,

    )


# def precess_arguments(arguments):



def main():
    parser = ArgumentParser(
        prog="core",
        description="Train and fit application.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    current_config_path = arguments.config_path
    if current_config_path != DEFAULT_CONFIG_PATH:
        current_config_path = "configs/" + current_config_path + ".yml"
    parametrs = read_training_pipeline_params(current_config_path)
    build_features(parametrs)
    fit_model(parametrs)
    main_predict(parametrs)


if __name__ == "__main__":
    main()
