import numpy as np
import pandas as pd
import requests
from json import dumps
from pprint import pprint

from src.enities.app_params import read_app_params


if __name__ == "__main__":
    parametrs = read_app_params()
    data = pd.read_csv(parametrs.data_for_predict_path)
    data["idx"] = range(data.shape[0])
    data = data.to_dict("records")
    response = requests.get(parametrs.url_external, data=dumps(data))
    print("Predicts:")
    pprint(response.json())
