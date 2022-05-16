import numpy as np
import sktime
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, make_scorer
from sklearn.metrics import r2_score

def mean_absolute_percentage_error_100(y, y_pred):
    Score_model = mean_absolute_percentage_error(y, y_pred)
    return 100*Score_model

def symmetric_mean_absolute_percentage_error(y, y_pred):
    smape = MeanAbsolutePercentageError(symmetric=True)
    Score_model = smape(y, y_pred)
    return 100*Score_model

def bias(y, y_pred):
    Score_model = y_pred.sum()/y.sum()
    return Score_model

def correlation_coefficient(y, y_pred):
    Score_model = np.corrcoef(y, y_pred)
    return Score_model[0,1]


def get_metric_name_mapping():
    return {_mape(): mean_absolute_percentage_error_100,
            _r2(): r2_score,
            _pearson(): correlation_coefficient,
            _smape():symmetric_mean_absolute_percentage_error,
            _bias(): bias   
    }


def get_metric_function(name: str, **params):
    mapping = get_metric_name_mapping()

    def fn(y, y_pred):
        return mapping[name](y, y_pred, **params)

    return fn


def get_scoring_function(name: str, **params):
    mapping = {
        _mape(): make_scorer(mean_absolute_percentage_error_100, greater_is_better=False, **params),
        _r2(): make_scorer(r2_score, **params),
        _pearson(): make_scorer(correlation_coefficient, **params),
        _smape(): make_scorer(symmetric_mean_absolute_percentage_error, **params),
        _bias(): make_scorer(bias, **params)
    }
    return mapping[name]


def _mape():
    return "mean absolute percentage error"

def _r2():
    return "r2"

def _pearson():
    return "pearson correlation coefficient"

def _smape():
    return "symmetric mean absolute percentage error"

def _bias():
    return "bias"

