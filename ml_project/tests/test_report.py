import os.path

from src.core import DEFAULT_CONFIG_PATH
from src.enities.all_train_params import read_training_pipeline_params


def test_file_exist():
    parametrs = read_training_pipeline_params(DEFAULT_CONFIG_PATH)
    path_to_dataset = parametrs.input_data_path
    assert os.path.exists(path_to_dataset) and os.path.isfile(path_to_dataset), (
        f"{path_to_dataset}"
    )
