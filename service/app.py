import sys
import typing as t
from datetime import datetime
from functools import lru_cache

import os
import joblib
import pandas as pd
from fastapi import FastAPI, Depends, Body  # type: ignore # noqa: E402
from pydantic import BaseSettings, PositiveFloat


from entities import ModelInput

app = FastAPI(title="API Crypto-Forecasting", version="0.0.1")


class Settings(BaseSettings):
    serialized_model_path: str
    serialized_prepro_path: str
    model_lib_dir: str

    class Config:
        env_file = './service/.env'
        env_file_encoding = 'utf-8'


@lru_cache(None)
def get_settings():
    return Settings()


@lru_cache(None)
def load_estimator():
    sys.path.append(get_settings().model_lib_dir)
    estimator = joblib.load(get_settings().serialized_model_path)
    return estimator

@lru_cache(None)
def load_prepro():
    sys.path.append(get_settings().model_lib_dir)
    estimator = joblib.load(get_settings().serialized_prepro_path)
    return estimator

class Logger:
    def __init__(self, file: t.TextIO = sys.stdout):
        self.file = file

    def log(self, inputs: t.List[ModelInput]):
        for row in inputs:
            record = {"datetime": datetime.now(), "input": row.dict()}
            print(record, file=self.file)


def get_logger():
    return Logger()


@app.post("/", response_model=t.List[float])
async def make_prediction(
    inputs: t.List[ModelInput] = Body(...),
    estimator=Depends(load_estimator),
    preprocessing = Depends(load_prepro),
    logger=Depends(get_logger),
):
    logger.log(inputs)
    X = pd.DataFrame([row.dict() for row in inputs])
    preprocessing_data = preprocessing.transform(X)
    steps = preprocessing_data.shape[0]
    prediction = estimator.predict(steps= steps, exog=preprocessing_data).tolist()
    return prediction


@app.get("/")
async def service_status():
    """Check the status of the service"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
