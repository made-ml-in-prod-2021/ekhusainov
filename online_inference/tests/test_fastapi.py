from json import dumps
import pytest

import numpy as np
import pandas as pd

from src.fastapi_app import app, check_request


from fastapi.testclient import TestClient

BAD_REQUEST = {"fail": [1, 2, 3]}
DEFAULT_X_TEST_PATH = "data/validate_part/x_test.csv"
GOOD_REQUEST = [{1: 2}]
HTTP_BAD_REQUEST = 400
HTTP_OK = 200


@pytest.fixture
def client():
    with TestClient(app) as current_client:
        yield current_client


@pytest.fixture()
def data():
    return pd.read_csv(DEFAULT_X_TEST_PATH)


def test_read_main(client):
    response = client.get("/")
    message = response.json()
    assert response.status_code == HTTP_OK
    assert "point" in message, (
        f"FAIL: {repr(message)}"
    )


def test_check_request(client):
    answer_1 = check_request(BAD_REQUEST)
    answer_2 = check_request(GOOD_REQUEST)
    assert answer_1 == False and answer_2 == True, (
        f"Wrong answer {answer_1} or {answer_2}"
    )


def test_correct_request(client, data):
    data = data.to_dict("records")
    response = client.post("/predict/", data=dumps(data))
    assert response.status_code == HTTP_OK, (
        f"Failed data: {dumps(data)}"
    )


def test_bad_request(client, data):
    data = data.iloc[:, -3:]
    data = data.to_dict("records")
    response = client.post("/predict/", data=dumps(data))
    assert response.status_code == HTTP_BAD_REQUEST
