import numpy as np
import pandas as pd
import requests
from json import dumps
from pprint import pprint

HTTP_PATH = "http://127.0.0.1:8000/predict/"

DEFAULT_X_TEST_PATH = "data/validate_part/x_test.csv"

if __name__ == "__main__":
    data = pd.read_csv(DEFAULT_X_TEST_PATH)
    data["idx"] = range(data.shape[0])
    data = data.to_dict("records")
    response = requests.get(HTTP_PATH, data=dumps(data))
    print("Predicts:")
    pprint(response.json()[:5])