"""Here we just generate a report in html"""
import logging
import logging.config
import pandas as pd

import yaml

from pandas_profiling import ProfileReport


APPLICATION_NAME = "creating_report"
OUTPUT_REPORT_HTML = "profile_report.html"
PATH_TO_DATASET = "../data/raw/heart.csv"
REPORT_LOGGING_CONFIG_FILEPATH = "../configs/report_logging.conf.yml"

logger = logging.getLogger(APPLICATION_NAME)


def setup_logging():
    with open(REPORT_LOGGING_CONFIG_FILEPATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


def read_csv_file(filepath: str) -> pd.DataFrame:
    logger.debug("Start reading the file.")
    data = pd.read_csv(filepath)
    logger.info("File %s was read", repr(filepath))
    return data


def creating_report_using_profile_report(input_data: str,
                                         output_filepath: str):
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
