import json
import pandas as pd
import numpy as np

from app import app
from app import HeartFeaturesModel


from fastapi.testclient import TestClient

DEFAULT_X_TEST_PATH = "data/validate_part/x_test.csv"

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    message = response.json()
    assert response.status_code == 200
    assert "point" in message, (
        f"FAIL: {repr(message)}"
    )


def test_simple_predict():
    data = pd.read_csv(DEFAULT_X_TEST_PATH)
    row_number = data.shape[0]
    data["idx"] = range(data.shape[0])
    data = data.to_dict("records")

    response = client.get("/predict/", data=json.dumps(data))
    predicts = response.json()

    assert response.status_code == 200, (
        print(f"FAIL: {response}")
    )
    assert row_number == len(predicts), (
        print(f"Wrong predict structure: {repr(predicts)}")
    )
