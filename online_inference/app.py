import logging
import os
from joblib import load
from typing import List, Union, Optional

import numpy as np
import pandas as pd
from pydantic.main import BaseModel
import uvicorn
from fastapi import FastAPI
from pydantic import BaseConfig, conlist
from sklearn.linear_model import LogisticRegression

from src.fit_predict.batch_predict import batch_predict_command

NUMBER_FEATURES = 13


class HeartFeaturesModel(BaseModel):
    data: List[conlist(Union[int, float],
                       min_items=NUMBER_FEATURES, max_items=NUMBER_FEATURES)]
    features: List[str]


class TargetResponse(BaseModel):
    id: str
    value: int


def make_predict(data: List,
                 features: List[str],
                 ) -> List[TargetResponse]:
    data = pd.DataFrame(data, columns=features)
    number_raw = data.shape[0]
    ids = list(range(number_raw))
    predicts = batch_predict_command(data)
    return [
        TargetResponse(id=id_, target=int(target)) for id_, target in zip(ids, predicts)
    ]


app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


# @app.on_event("startup")
# def load_model():
#     global mode

@app.get("/predict/", response_model=List[TargetResponse])
def predict(request: HeartFeaturesModel):
    return make_predict(request.data, request.features)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
