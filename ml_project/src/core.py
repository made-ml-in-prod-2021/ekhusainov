"Our main module."
from argparse import (
    ArgumentParser,
    ArgumentDefaultsHelpFormatter,
)
from src.enities.all_train_params import (
    read_training_pipeline_params,
)
from src.features.build_features import build_features
from src.fit_predict.fit_model import fit_model
from src.fit_predict.predict import main_predict


DEFAULT_CONFIG_NAME = "logregr"
DEFAULT_CONFIG_PATH = "configs/logregr.yml"


def setup_parser(parser):
    "Argparser."
    parser.add_argument(
        "-c", "--config",
        help="the name of the file located in configs/, without .yml, \
            default name is \"%(default)s\"",
        dest="config_name",
        default=DEFAULT_CONFIG_NAME,
    )


def process_arguments(config_name: str):
    "Parse config."
    current_config_path = config_name
    current_config_path = "configs/" + current_config_path + ".yml"
    parametrs = read_training_pipeline_params(current_config_path)
    build_features(parametrs)
    fit_model(parametrs)
    main_predict(parametrs)


def main():
    "Our int main."
    parser = ArgumentParser(
        prog="core",
        description="Train and fit application.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    process_arguments(arguments.config_name)


if __name__ == "__main__":
    main()
