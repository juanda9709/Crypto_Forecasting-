import os
from pickle import FALSE
import shutil
import typing as t
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from sqlalchemy import column

import typer
import yaml
import joblib
from sklearn.base import BaseEstimator
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
import warnings

import data
import model
import metrics

app = typer.Typer()


@lru_cache(None)
def _read_csv(filepath):
    return pd.read_csv(filepath)


class CsvDatasetReader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def __call__(self):
        return _read_csv(self.filepath)


@app.command()
def train(config_file: str):
    config_file = "config.yml"
    hyperparams = _load_config(config_file, "hyperparams")
    regressor = _load_config(config_file, "regressor")
    split = "train"
    X, y = _get_dataset(_load_config(config_file, "data"), splits=[split])[split]
    estimator = model.build_estimator(hyperparams)
    preprocessing_data = estimator.fit_transform(X,y)
    forecaster = model.build_estimator_(regressor)
    forecaster.regressor

    forecaster.fit(y = y, exog=preprocessing_data)
    output_dir = _load_config(config_file, "export")["output_dir"]
    version = _save_versioned_estimator(estimator, hyperparams, forecaster, regressor, output_dir)
    return version


def _get_dataset(data_config, splits):
    filepath = data_config["filepath"]
    reader = CsvDatasetReader(filepath)
    return data.get_dataset(reader=reader, splits=splits, 
                            days=data_config["days"], test_size=data_config["test_size"],
                            crypto=data_config["crypto"]
                            )


def _save_versioned_estimator(
    estimator: BaseEstimator, hyperparams: t.Dict[str, t.Any],
    forecaster: BaseEstimator, regressor: t.Dict[str, t.Any],
    output_dir: str
):
    version = str(datetime.now(timezone.utc).replace(second=0, microsecond=0))
    model_dir = os.path.join(output_dir, version) 
    model_dir1 = model_dir.replace(":", "_")
    model_dir1 = model_dir1.replace(" ", "_") 
    model_dir1 = model_dir1 + "_pre"
    model_dir2 = model_dir.replace(":", "_")
    model_dir2 = model_dir2.replace(" ", "_") 
    model_dir2 = model_dir2 + "_for"
    os.makedirs(model_dir1, exist_ok=True)
    os.makedirs(model_dir2, exist_ok=True)
    try:
        joblib.dump(estimator, os.path.join(model_dir1, "model.joblib"))
        _save_yaml(hyperparams, os.path.join(model_dir1, "params.yml"))
    except Exception as e:
        typer.echo(f"Coudln't serialize model due to error {e}")
        shutil.rmtree(model_dir1)
    try:
        joblib.dump(forecaster, os.path.join(model_dir2, "model.joblib"))
        _save_yaml(regressor, os.path.join(model_dir2, "params.yml"))
    except Exception as e:
        typer.echo(f"Coudln't serialize model due to error {e}")
        shutil.rmtree(model_dir2)
    return version


@app.command()
def find_hyperparams(
    config_file: str,
    train_best_model: bool = typer.Argument(False),
):
    search_config = _load_config(config_file, "search")
    prepro = search_config["preprocessing"]
    reg = search_config["regressor"]
    n_jobs = search_config["jobs"]
    metric = _load_config(config_file, "metrics")[0]
    dummy_preprocessing = {name: {} for name in prepro.keys()}
    regressor = {name: {} for name in reg.keys()}    
    scoring = metrics.get_scoring_function(metric["name"], **metric["params"])
    param_grid = search_config["regressor_grid"]
    lags_grid = search_config["forecaster_grid"]["lags"]
    estimator = model.build_estimator(dummy_preprocessing)
    forecaster = model.build_estimator_(regressor)
    split = "train"
    X, y = _get_dataset(_load_config(config_file, "data"), splits=[split])[split]
    preprocessing_data = estimator.fit_transform(X,y)
    cv = TimeSeriesSplit(n_splits=5)
    gs = grid_search_forecaster(
        forecaster = forecaster,
        y = y,
        exog = preprocessing_data,
        param_grid= _param_grid_to_sklearn_format(param_grid),
        lags_grid= lags_grid,
        steps=15,
        metric= MeanAbsolutePercentageError(symmetric=True),
        return_best=True,

        initial_train_size = int(y.shape[0] - y.shape[0]*0.1),         
    )
    #_param_grid_to_sklearn_format(param_grid),
    #gs.fit(X, y)
    hyperparams = forecaster.regressor[0].get_params()
    output_dir = _load_config(config_file, "export")["output_dir"]
    _save_versioned_estimator(estimator, dummy_preprocessing , forecaster, hyperparams, output_dir)


def _param_grid_to_sklearn_format(param_grid):
    return {
        f"{name}__{pname}": pvalues
        for name, params in param_grid.items()
        for pname, pvalues in params.items()
    }


def _param_grid_to_custom_format(param_grid):
    grid = {}
    for name, values in param_grid.items():
        estimator_name, param_name = name.split("__", maxsplit=1)
        if estimator_name not in grid:
            grid[estimator_name] = {}
        grid[estimator_name][param_name] = values
    return grid

@app.command()
def eval(
    config_file: str,
    prepro_version: str,
    model_version: str,    
    splits: t.List[str] = ["test"],
):
    output_dir = _load_config(config_file, "export")["output_dir"]
    saved_model = os.path.join(output_dir, model_version, "model.joblib")
    prepro_model = os.path.join(output_dir, prepro_version, "model.joblib")
    estimator = joblib.load(saved_model)
    prepro = joblib.load(prepro_model)
    dataset = _get_dataset(_load_config(config_file, "data"), splits=splits)
    report = defaultdict(list)

    all_metrics = _load_config(config_file, "metrics")
    for name, (X, y) in dataset.items():
        preprocessing_data = prepro.transform(X)
        pasos = X.shape[0]
        y_pred = estimator.predict(steps=pasos, exog=preprocessing_data)

        for m in all_metrics:
            metric_name, params = m["name"], m["params"]
            fn = metrics.get_metric_function(metric_name, **params)
            value = float(fn(y, y_pred))
            report[metric_name].append({"split": name, "value": value})
    reports_dir = _load_config(config_file, "reports")["dir"]
    _save_yaml(
        dict(report),
        os.path.join(reports_dir, f"{model_version}.yml"),
    )
    


def _load_config(filepath: str, key: str):
    content = _load_yaml(filepath)
    config = content[key]
    return config


@lru_cache(None)
def _load_yaml(filepath: str) -> t.Dict[str, t.Any]:
    with open(filepath, "r") as f:
        content = yaml.safe_load(f)
    return content


def _save_yaml(content: t.Dict[str, t.Any], filepath: str):
    with open(filepath, "w") as f:
        yaml.dump(content, f)


if __name__ == "__main__":
    
   # app(["train", "config_file"])
   # app(["find-hyperparams", "config.yml"])
  #  app(["eval", "config.yml", "2022-06-02_05_26_00+00_00"])
   
    app()
