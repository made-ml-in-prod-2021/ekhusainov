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
from src.enities.app_params import read_app_params

NUMBER_FEATURES = 13


class HeartFeaturesModel(BaseModel):
    age: float = 55.16
    sex: int = 1
    cp: int = 1
    trestbps: float = 137.26
    chol: float = 243.14
    fbs: int = 0
    restecg: int = 0
    thalach: float = 140.38
    exang: int = 0
    oldpeak: float = 1.07
    slope: int = 1
    ca: int = 1
    thal: int = 3
    idx: int = 0


class TargetResponse(BaseModel):
    idx: int
    value: int


def make_predict(data: List,
                 features: List[str],
                 ) -> List[TargetResponse]:
    idx_list = data["idx"].tolist()
    data.drop(["idx"], axis=1, inplace=True)
    predicts = batch_predict_command(data)
    answer = []
    for i, target in zip(idx_list, predicts.to_numpy()):
        answer.append(TargetResponse(idx=i, value=target))
    return answer


app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.get("/predict/", response_model=List[TargetResponse])
def predict(request: List[HeartFeaturesModel]):
    data = pd.DataFrame(element.__dict__ for element in request)
    features = data.columns.tolist()
    answer = make_predict(data, features)
    return answer


if __name__ == "__main__":
    parametrs = read_app_params()
    uvicorn.run("app:app", host=parametrs.ip_inside_docker,
                port=parametrs.port)
