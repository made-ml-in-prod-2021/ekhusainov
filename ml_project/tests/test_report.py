import os.path

import pytest
from report.creating_report import PATH_TO_DATASET



def test_file_exist():
    current_dataset_path = PATH_TO_DATASET[3:]
    assert os.path.exists(current_dataset_path) and os.path.isfile(current_dataset_path), (
        f"{current_dataset_path}"
    )
