"""Machine learning in production. HW02."""
from typing import List, Tuple
import logging
import logging.config
import os

from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic.main import BaseModel
import pandas as pd
import uvicorn


from src.enities.app_params import read_app_params
from src.enities.logging_params import setup_logging
from src.fit_predict.batch_predict import batch_predict_command

APPLICATION_NAME = "fastapi_app"
HTTP_BAD_REQUEST = 400
HTTP_OK = 200
LOCAL_PATH_CONFIG = "models/config.joblib"
NUMBER_FEATURES = 13

logger = logging.getLogger(APPLICATION_NAME)
setup_logging()

models_tuple: Tuple = (None, None, None)  # model, one_hot, scaler


class TargetResponse(BaseModel):
    """ID and predict."""
    idx: int = 0
    value: int = 0


def make_predict(models_tuple: Tuple,
                 data: List,
                 features: List[str],
                 ) -> List[TargetResponse]:
    logger.info("Begin read predict data.")
    data = pd.DataFrame(data, columns=features)
    idx_list = list(range(data.shape[0]))
    predicts = batch_predict_command(models_tuple, data)
    logger.info("Finish predict data.")

    answer = []
    for i, target in zip(idx_list, predicts.to_numpy()):
        answer.append(TargetResponse(idx=i, value=target))
    return answer


app = FastAPI()


def check_request(request: List[dict]):
    """If false -> HTTP 400."""
    if not isinstance(request, list):
        return False
    for elem in request:
        if not isinstance(elem, dict):
            return False
        for value in elem.values():
            if not isinstance(value, int) and not isinstance(value, float):
                return False
    return True


@app.get("/")
def root_path():
    """Our root path with button."""
    return "It is entry point of our predictor."


@app.on_event("startup")
def load_trained_model(config_path: str = LOCAL_PATH_CONFIG):
    global models_tuple

    if not os.path.exists(config_path):
        error_msg = "No config."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    parametrs = load(config_path)
    model_path = parametrs.output_model_path
    one_hot_code_path = parametrs.path_to_one_hot_encoder
    scale_model_path = parametrs.path_to_scaler

    if not os.path.exists(model_path):
        error_msg = "No model."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    if not os.path.exists(one_hot_code_path):
        error_msg = "No one hot enconder."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    if not os.path.exists(scale_model_path):
        error_msg = "No scale model."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    model = load(model_path)
    one_hot_code_model = load(one_hot_code_path)
    scale_model = load(scale_model_path)

    models_tuple = tuple([
        model,
        one_hot_code_model,
        scale_model,
    ])


@app.post("/predict/", response_model=List[TargetResponse])
def predict(request: List[dict]):
    """
    Getting the database and returning the predictions.
    """
    if not check_request(request):
        error_massage = "Wrong type."
        logger.error(error_massage)
        raise HTTPException(
            detail=error_massage,
            status_code=HTTP_BAD_REQUEST,
        )
    features = list(request[0].keys())
    data = [list(elem.values()) for elem in request]
    return make_predict(models_tuple, data, features)


def main():
    """Our int main."""

    parametrs = read_app_params()
    uvicorn.run(app, host=parametrs.ip_inside_docker,
                port=parametrs.port)


if __name__ == "__main__":
    main()
