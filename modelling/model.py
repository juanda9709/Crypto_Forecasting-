import typing as t

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def build_estimator(hyperparams: t.Dict[str, t.Any]):
    estimator_mapping = get_estimator_mapping()
    steps = []
    for name, params in hyperparams.items():
        estimator = estimator_mapping[name](**params)
        steps.append((name, estimator))
    model = Pipeline(steps)
    return model


def get_estimator_mapping():
    return {
        "regressor": RandomForestRegressor,
        "age-extractor": AgeExtractor,
        "column-transformer": CustomColumnTransformer,
        "simplified-transformer": SimplifiedTransformer,
        "reindexar": Reindexar,
       # "select-crypto": SelectCrypto,
    }


class Reindexar(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        return X.reindex(range(X.index[0],X.index[-1]+60,60),method='pad')


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


'''

class SelectCrypto(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  
    def transform(self, X, y):

        df = df[df["Asset_ID"]==1].set_index("timestamp") 
        return df 
'''

class AgeExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
        X["RemodAddAge"] = X["YrSold"] - X["YearRemodAdd"]
        X["GarageAge"] = X["YrSold"] - X["GarageYrBlt"]
        return X


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
