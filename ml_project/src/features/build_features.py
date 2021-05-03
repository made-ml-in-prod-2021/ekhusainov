"""Preparing data for training."""
import logging
import logging.config
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib

import yaml


APPLICATION_NAME = "build_features"
PATH_TO_DATASET = "../../data/raw/heart.csv"
REPORT_LOGGING_CONFIG_FILEPATH = "../../configs/train_model_logging.conf.yml"
PATH_TO_ONE_HOT_ENCODER = "../../models/one_hot.joblib"

logger = logging.getLogger(APPLICATION_NAME)


def setup_logging():
    "Logger from yaml config."
    with open(REPORT_LOGGING_CONFIG_FILEPATH) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


def read_csv_file(filepath: str) -> pd.DataFrame:
    "Read raw data."
    logger.debug("Start reading the file.")
    data = pd.read_csv(filepath)
    logger.info("File %s was read", repr(filepath))
    return data


def split_to_train_test(data: pd.DataFrame, test_size=0.15) -> tuple:
    "Split raw data to x_train, x_test, y_train, y_test."
    logger.debug("Start to split datatest to train and test.")
    x_data = data.drop(['target'], axis=1)
    target = data['target'].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, target, test_size=test_size, random_state=1337, stratify=target)
    logger.info("Finish split datatest to train and test.")
    return x_train, x_test, y_train, y_test


def split_dataset_to_num_cat_features(x_data: pd.DataFrame) -> tuple:
    "One data to tuple (categorial_data, num_data)."
    logger.debug("Start to split dataset to numeric and categorial features")
    columns_x_data = x_data.columns.tolist()
    if "target" in columns_x_data:
        logger.info("The full dataset with \"target\" is given for input")
        x_data = x_data.drop(['target'], axis=1)
    cat_columns = [
        "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
    ]
    num_columns = [
        "age", "trestbps", "chol", "thalach", "oldpeak",
    ]
    categorial_data = x_data[cat_columns]
    numeric_data = x_data[num_columns]
    logger.info("""Finished dividing the dataset into categorical and \
    numeric variables""")
    return categorial_data, numeric_data


def categorial_feature_to_one_hot_encoding(
        categorial_data: pd.DataFrame,
        filepath=PATH_TO_ONE_HOT_ENCODER) -> np.array:
    "Transform categorial features to one hot encoding and safe model"
    logger.debug("Start to one hot encoding.")
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore")
    one_hot_encoder.fit(categorial_data)
    transformed_to_one_hot = one_hot_encoder.transform(
        categorial_data).toarray()
    logger.info("Finish one hot encoding.")
    logger.debug("Start to save one hot model")
    joblib.dump(one_hot_encoder, filepath)
    logger.info("Finish to save one hot model")
    return transformed_to_one_hot


def main():
    "Our int main."
    setup_logging()
    data = read_csv_file(PATH_TO_DATASET)


if __name__ == "__main__":
    main()
