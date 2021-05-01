"""Here we just generate a report in html"""
import logging
import logging.config
import pandas as pd

import yaml

from pandas_profiling import ProfileReport

APPLICATION_NAME = "creating_report"
DEFAULT_LOGGING_CONFIG_FILEPATH = "logging.conf.yml"
OUTPUT_REPORT_HTML = "profile_report.html"
PATH_TO_DATASET = "../data/raw/heart.csv"

logger = logging.getLogger(APPLICATION_NAME)


def setup_logging():
    with open(DEFAULT_LOGGING_CONFIG_FILEPATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))

    # simple_formatter = logging.Formatter(
    #     fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    # )

    # file_handler = logging.FileHandler(
    #     filename="report.log",
    # )
    # file_handler.setLevel(logging.DEBUG)
    # file_handler.setFormatter(simple_formatter)

    # logger = logging.getLogger(APPLICATION_NAME)
    # logger.setLevel(logging.DEBUG)
    # logger.addHandler(file_handler)

    # logger = logging.getLogger()
    # logger.addHandler(file_handler)


def read_csv_file(filepath):
    logger.debug("Start reading the file.")
    data = pd.read_csv(filepath)
    logger.info("File %s was read", repr(filepath))
    return data


def creating_report_using_profile_report(input_data, output_filepath):
    logger.debug("The report begins to be written.")
    profile = ProfileReport(input_data)
    logger.info("The report is ready.")
    profile.to_file(output_file=output_filepath)
    logger.info("The report is saved to a file %s.", repr(output_filepath))


def main():
    setup_logging()
    data = read_csv_file(PATH_TO_DATASET)
    creating_report_using_profile_report(data, OUTPUT_REPORT_HTML)


if __name__ == "__main__":
    main()
