from json import dumps
import pytest

import numpy as np
import pandas as pd

from src.fastapi_app import app, check_request


from fastapi.testclient import TestClient

DEFAULT_X_TEST_PATH = "data/validate_part/x_test.csv"
HTTP_BAD_REQUEST = 400
HTTP_OK = 200

client = TestClient(app)


@pytest.fixture()
def data():
    return pd.read_csv(DEFAULT_X_TEST_PATH)


def test_read_main():
    response = client.get("/")
    message = response.json()
    assert response.status_code == HTTP_OK
    assert "point" in message, (
        f"FAIL: {repr(message)}"
    )


def test_check_request(data):
    answer_1 = check_request({"fail": [1, 2, 3]})
    answer_2 = check_request([{1: 2}])
    assert answer_1 == False and answer_2 == True, (
        f"Wrong answer {answer_1} or {answer_2}"
    )


def test_correct_request(data):
    data = data.to_dict("records")
    response = client.get("/predict/", data=dumps(data))
    assert response.status_code == HTTP_OK


def test_bad_request(data):
    data = data.iloc[:, -3:]
    data = data.to_dict("records")
    response = client.get("/predict/", data=dumps(data))
    assert response.status_code == HTTP_BAD_REQUEST
