import numpy as np
import pandas as pd

from src.enities.all_train_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from src.core import DEFAULT_CONFIG_PATH
from src.features.build_features import build_features
from src.fit_predict.fit_model import fit_model
from src.fit_predict.predict import main_predict

FAKE_DATASET_SIZE = 100
DEFAULT_FAKE_DATA_PATH = "data/fake_data/fake_data.csv"


def test_generate_fake_data():
    parametrs = read_training_pipeline_params(DEFAULT_CONFIG_PATH)
    data = pd.read_csv(parametrs.input_data_path)
    data_fake = data.drop(data.index)
    cat_columns = parametrs.features_params.categorial_features
    num_columns = parametrs.features_params.numerical_features
    for column in num_columns:
        column_random = np.random.normal(
            loc=data[column].mean(),
            scale=data[column].std(),
            size=FAKE_DATASET_SIZE,
        )
        data_fake[column] = column_random
    target = parametrs.features_params.target_column
    cat_columns = cat_columns + [target]
    for column in cat_columns:
        count_cat_value = data[column].value_counts(normalize=True)
        cat_random = np.random.choice(
            count_cat_value.index,
            size=FAKE_DATASET_SIZE,
            p=count_cat_value.values,
        )
        data_fake[column] = cat_random
    data_fake.to_csv(DEFAULT_FAKE_DATA_PATH, index=False)
    parametrs.input_data_path = DEFAULT_FAKE_DATA_PATH
    build_features(parametrs, on_logger=False)
    fit_model(parametrs)
    ac_score = main_predict(parametrs)
    fake_data_generation_passed = bool(ac_score)
    assert fake_data_generation_passed, (
        f"Fake_ac_score = {ac_score}"
    )
