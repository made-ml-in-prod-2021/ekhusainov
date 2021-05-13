"""Send data to the ip and get predictions."""
import logging
from json import dumps
from pprint import pprint
import requests


import pandas as pd


from src.enities.app_params import (
    read_app_params,
    AppParams,
)
from src.enities.logging_params import setup_logging

APPLICATION_NAME = "make_request"
HTTP_BAD_REQUEST = 400
HTTP_OK = 200

logger = logging.getLogger(APPLICATION_NAME)


def simple_response(data: pd.DataFrame,
                    parametrs: AppParams) -> None:
    """Makes response."""
    data = data.to_dict("records")
    response = requests.get(parametrs.url_external,
                            data=dumps(data))
    
    print(f"HTTP:{response.status_code}")
    print("Predicts:")
    pprint(response.json())
    if response.status_code == HTTP_OK:
        for line in response.json():
            logger.info("%s", repr(line))


def main():
    """Our int main"""
    setup_logging()
    parametrs = read_app_params()
    data = pd.read_csv(parametrs.data_for_predict_path)

    simple_response(data, parametrs)
    simple_response(data.iloc[:, -3:], parametrs)


if __name__ == "__main__":
    main()
