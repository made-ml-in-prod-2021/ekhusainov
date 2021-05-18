"""Here we just generate a report in html"""
import logging
import logging.config

import pandas as pd
from pandas_profiling import ProfileReport

import yaml

from src.enities.all_train_params import (
    read_training_pipeline_params,
    TrainingPipelineParams,
)
from src.core import CONFIG_PATH


APPLICATION_NAME = "creating_report"
REPORT_LOGGING_CONFIG_FILEPATH = "configs/report_logging.conf.yml"

logger = logging.getLogger(APPLICATION_NAME)


def setup_logging():
    """Logger from yaml config."""
    with open(REPORT_LOGGING_CONFIG_FILEPATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


def read_csv_file(parametrs: TrainingPipelineParams) -> pd.DataFrame:
    """Read raw data."""
    filepath = parametrs.input_data_path
    logger.debug("Start reading the file.")
    data = pd.read_csv(filepath)
    logger.info("File %s was read", repr(filepath))
    return data


def creating_report_using_profile_report(input_data: pd.DataFrame,
                                         parametrs: TrainingPipelineParams):
    """Create report and save to a html file."""
    output_filepath = parametrs.output_report_html
    logger.debug("The report begins to be written.")
    profile = ProfileReport(input_data)
    logger.info("The report is ready.")
    profile.to_file(output_file=output_filepath)
    logger.info("The report is saved to a file %s.", repr(output_filepath))


def main():
    """Out int main."""
    setup_logging()
    parametrs = read_training_pipeline_params(CONFIG_PATH)
    data = read_csv_file(parametrs)
    creating_report_using_profile_report(data, parametrs)


if __name__ == "__main__":
    main()
