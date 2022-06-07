import typing as t
import typing_extensions as te

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit


class DatasetReader(te.Protocol):
    def __call__(self) -> pd.DataFrame:
        ...


SplitName = te.Literal["train", "test"]


def get_dataset(reader: DatasetReader, 
    splits: t.Iterable[SplitName],
    days: int,
    test_size: float,
    crypto: int
):
    total_minutes = days*1440
    train_minutes = int(total_minutes*(1-test_size))
    test_minutes = total_minutes - train_minutes

    df = reader()
    df = clean_dataset(df, crypto)

    X_train, y_train = _split_train(df, test_minutes, total_minutes)
    X_test, y_test = _split_test(df, test_minutes)


    split_mapping = {"train": (X_train, y_train), "test": (X_test, y_test)}
    return {k: split_mapping[k] for k in splits}


def clean_dataset(df: pd.DataFrame, crypto: int) -> pd.DataFrame:
    cleaning_fn = _chain(
        [   _fill_NaN,
            _select_crypto,
            _reindexar
        ],
        crypto
    )
    df = cleaning_fn(df, crypto)
    return df


def _chain(functions: t.List[t.Callable[[pd.DataFrame], pd.DataFrame]], crypto: int):
    def helper(df, crypto):
        for fn in functions:
            df = fn(df, crypto)
        return df

    return helper

def _fill_NaN(df, crypto):
    df = df.fillna(0)
    return df

def _reindexar (df, cryto):
    return df.reindex(range(df.index[0],df.index[-1]+60,60),method='pad')

def _select_crypto(df, crypto):

    df = df[df["Asset_ID"]==crypto].set_index("timestamp") 
    return df

def _split_train(df,test_minutes, total_minutes):
    y = df.Target
 
    X_train = df.iloc[-1*total_minutes:-1*test_minutes]
    X_train = X_train.drop(['Target'], axis = 1)
    y_train = y.iloc[-1*total_minutes:-1*test_minutes]

    return X_train, y_train


def _split_test(df, test_minutes):
    y = df.Target


    X_test = df.iloc[-1*test_minutes:-16]
    X_test = X_test.drop(['Target'], axis = 1)
    y_test = y.iloc[-1*test_minutes:-16]

    return X_test, y_test

    
