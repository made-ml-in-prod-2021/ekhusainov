import os.path

from report.creating_report import PATH_TO_DATASET



def test_file_exist():
    assert os.path.exists(PATH_TO_DATASET) and os.path.isfile(PATH_TO_DATASET), (
        f"{PATH_TO_DATASET}"
    )
