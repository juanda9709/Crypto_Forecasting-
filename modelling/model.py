from tarfile import ExtractError
import typing as t
import pandas as pd
import numpy as np
import skforecast
import lightgbm as lgbm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tsfresh import extract_features,select_features
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.utilities.dataframe_functions import roll_time_series
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.svm import LinearSVR

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster


def build_estimator(hyperparams: t.Dict[str, t.Any]):
    estimator_mapping = get_estimator_mapping()
    steps = []
    for name, params in hyperparams.items():
        estimator = estimator_mapping[name](**params)
        steps.append((name, estimator))
    model = Pipeline(steps)
    return model

def build_estimator_(hyperparams: t.Dict[str, t.Any]):
    estimator_mapping = get_estimator_mapping()
    steps = []
    for name, params in hyperparams.items():
        estimator = estimator_mapping[name](**params)
        steps.append((name, estimator))
    model = Pipeline(steps)
    forecaster = ForecasterAutoreg(regressor = model, lags = 3)
    return forecaster

def get_estimator_mapping():
    return {
        "RandomForest": RandomForestRegressor,
        "KnnRegressor": KNeighborsRegressor,
        "LGBMRegressor": lgbm.LGBMRegressor,
        "SGDRegressor": SGDRegressor,
        "RidgeRegressor": Ridge,
        "SVRRegressor": LinearSVR,        
        "tsfresh-rolled-data": TsRolledData,
        "tsextractFeatures": ExtractFeatures,
        "baseline-preprocessing": BasicTransformer,
        "basic-scaler": BasicScaler        
    }

class BasicScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_ = X.copy()
     #   scaler = MinMaxScaler()
        ind = X_.index
        scaler = StandardScaler()
        X_[list(X_.columns)] = scaler.fit_transform(X_)
        final_df = pd.DataFrame(X_, index=ind)
        final = final_df.reindex(range(final_df.index[0],final_df.index[-1]+60,60),method='pad')
        return final



class BasicTransformer(BaseEstimator, TransformerMixin):
    _float_columns = (
        "Count,Open,High,Low,Close,Volume,VWAP"
    ).split(",")

    _ignored_columns = "Asset_ID".split(",")

    def __init__(self):
        self._column_transformer = ColumnTransformer(
            transformers=[
                ("droper", "drop", type(self)._ignored_columns),
                ("scaler", StandardScaler(with_mean=False), type(self)._float_columns),
            ],
            remainder="passthrough",
        )

    def fit(self, X, y=None):
        self._column_transformer = self._column_transformer.fit(X, y=y)
        return self

    def transform(self, X):
        return self._column_transformer.transform(X)


class TsRolledData(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X_ = X.copy()
        X_['timstamp'] = X_.index
        df_rolled = roll_time_series(X_, column_id="Asset_ID", column_sort='timstamp', max_timeshift=15, min_timeshift=0)
        df_rolled.drop(['Asset_ID'], axis = 1, inplace=True)
        df_rolled['id']=df_rolled.apply(lambda x: x['id'][1],axis=1)

        return df_rolled

class ExtractFeatures(BaseEstimator, TransformerMixin):

    _kind_to_fc_parameters = None

    def __init__(self):
        _kind_to_fc_parameters = None

    def fit(self, X, y=None):
        parameters = MinimalFCParameters()
       # del parameters['friedrich_coefficients']
       # del parameters['max_langevin_fixed_point']
        df_features = extract_features(X, column_id="id", column_sort="timstamp",
                                default_fc_parameters=parameters
                                 )
        X_filtered = select_features(df_features, y.to_numpy().flatten())
        self._kind_to_fc_parameters = from_columns(X_filtered)
        return self

    def transform(self, X):
        df_final = extract_features(X, column_id="id", column_sort="timstamp",
                                kind_to_fc_parameters=self._kind_to_fc_parameters
                                )
        return df_final


