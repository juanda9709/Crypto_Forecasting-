from tarfile import ExtractError
import typing as t
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.utilities.dataframe_functions import roll_time_series



def build_estimator(hyperparams: t.Dict[str, t.Any]):
    estimator_mapping = get_estimator_mapping()
    steps = []
    for name, params in hyperparams.items():
        estimator = estimator_mapping[name](**params)
        steps.append((name, estimator))
    model = Pipeline(steps)
    # if 'tsextractFeatures' in list(model.named_steps.keys()):
    #     df_rolled_ = _df_rolled(X_)
    #     model.set_params(tsextractFeatures__timeseries_container=df_rolled_)
    return model

def get_estimator_mapping():
    return {
        "regressor": RandomForestRegressor,
        "tsfresh-rolled-data": TsRolledData,
        "column-transformer": CustomColumnTransformer,
        "simplified-transformer": SimplifiedTransformer,
        "tsextractFeatures": ExtractFeatures,
        "basic-scaler": BasicScaler
    }

class BasicScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_ = X.copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_)
        return X_scaled



class CustomColumnTransformer(BaseEstimator, TransformerMixin):
    _float_columns = (
        "Count,Open,High,Low,Close,Volume,VWAP"
    ).split(",")

    _ignored_columns = "Asset_ID".split(",")

    def __init__(self):
        self._column_transformer = ColumnTransformer(
            transformers=[
                ("droper", "drop", type(self)._ignored_columns),
                ("scaler", StandardScaler(), type(self)._float_columns),
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

    def transform(self, X):
        X = X.copy()
        X['timstamp'] = X.index
        df_rolled = roll_time_series(X, column_id="Asset_ID", column_sort='timstamp', max_timeshift=15, min_timeshift=0)
        df_rolled.drop(['Asset_ID'], axis = 1, inplace=True)
        df_rolled['id']=df_rolled.apply(lambda x: x['id'][1],axis=1)

        return df_rolled

class ExtractFeatures(BaseEstimator, TransformerMixin):

    _kind_to_fc_parameters = None

    def __init__(self):
        _kind_to_fc_parameters = None

    def fit(self, X, y=None):
        df_features = extract_features(X, column_id="id", column_sort="timstamp",
                                default_fc_parameters=MinimalFCParameters()
                                 )
        X_filtered = select_features(df_features, y.to_numpy().flatten())
        self._kind_to_fc_parameters = from_columns(X_filtered)
        return self

    def transform(self, X):
        df_final = extract_features(X, column_id="id", column_sort="timstamp",
                                kind_to_fc_parameters=self._kind_to_fc_parameters
                                )
        return df_final



class SimplifiedTransformer(BaseEstimator, TransformerMixin):
    """This is just for easy of demonstration"""

    _columns_to_keep = "HouseAge,GarageAge,LotArea,Neighborhood,HouseStyle".split(",")

    def __init__(self):
        self._column_transformer = ColumnTransformer(
            transformers=[
                ("binarizer", OrdinalEncoder(), ["Neighborhood", "HouseStyle"]),
            ],
            remainder="drop",
        )

    def fit(self, X, y=None):
        columns = type(self)._columns_to_keep
        X_ = X[columns]
        self._column_transformer = self._column_transformer.fit(X_, y=y)
        return self

    def transform(self, X):
        columns = type(self)._columns_to_keep
        X_ = X[columns]
        X_ = self._column_transformer.transform(X_)
        return X_

        

# class Reindexar(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X):
#         X = X.copy()
#         return X.reindex(range(X.index[0],X.index[-1]+60,60),method='pad')

# class TsRolledData(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         X__ = X.copy()
#         X__['timstamp'] = X__.index
#         X__ = X__.reset_index()
#         ind = pd.DataFrame(index = X__.index)
#         return ind

