"""Machine learning in production. HW02."""
from typing import List
import logging
import logging.config


from fastapi import FastAPI, HTTPException
from pydantic.main import BaseModel
import pandas as pd
import uvicorn


from src.enities.app_params import read_app_params
from src.enities.logging_params import setup_logging
from src.fit_predict.batch_predict import batch_predict_command

APPLICATION_NAME = "app"
HTTP_BAD_REQUEST = 400
HTTP_OK = 200
NUMBER_FEATURES = 13

logger = logging.getLogger(APPLICATION_NAME)


class TargetResponse(BaseModel):
    """ID and predict."""
    idx: int = 0
    value: int = 0


def make_predict(data: List,
                 features: List[str],
                 ) -> List[TargetResponse]:
    logger.info("Begin read predict data.")
    data = pd.DataFrame(data, columns=features)
    idx_list = list(range(data.shape[0]))
    predicts = batch_predict_command(data)
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
    return True


@ app.get("/")
def root_path():
    """Our root path with button."""
    # TODO button
    return "It is entry point of our predictor."


@ app.get("/predict/", response_model=List[TargetResponse])
def predict(request: List[dict]):
    if not check_request(request):
        error_massage = "Wrong type."
        logger.error(error_massage)
        raise HTTPException(
            detail=error_massage,
            status_code=HTTP_BAD_REQUEST,
        )
    features = list(request[0].keys())
    data = [list(elem.values()) for elem in request]
    return make_predict(data, features)


def main():
    """Our int main."""
    setup_logging()
    parametrs = read_app_params()
    uvicorn.run("app:app", host=parametrs.ip_inside_docker,
                port=parametrs.port)


if __name__ == "__main__":
    main()
