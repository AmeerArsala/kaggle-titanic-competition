import numpy as np
import pandas as pd

import tensorflow as tf

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from general_utils import alike_matches


def preprocessed_statistics(preprocessor, X):
    print("calling preprocessor.fit_transform()")
    X_ = preprocessor.fit_transform(X)
    print("Transformation complete.")

    # statistics after preprocessing

    print(f"type: {type(X_)}")

    shape = X_.shape
    dtype = X_.dtype

    print(f"shape: {shape}")
    print(f"dtype: {dtype}")

    return (X_, shape, dtype)


class AsNumPy(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.to_numpy(copy=True, dtype=np.float64)

        #if self.verbose:
            #print(X.shape)
            #print(X.dtypes)

            #print("|\nv")  # arrow

            #print(X_.shape)
            #print(X_.dtype)
            #print(type(X_))

        return X_


class WrapAsDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        super().__init__()
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.feature_names is None:
            return pd.DataFrame(X)

        return pd.DataFrame(X, columns=self.feature_names)


# assume operation on a pandas DataFrame
class DropFeatures(BaseEstimator, TransformerMixin):
    # cols: array of either indices or column/row names
    def __init__(self, cols):
        super().__init__()
        self.cols = cols
        self.use_column_names = (type(cols[0]) is str)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.use_column_names:
            return X.drop(self.cols, axis=1)
        else:
            return X.drop(X.columns[self.cols], axis=1)


# assume operation on a pandas DataFrame
class AddCluster(BaseEstimator, TransformerMixin):
    def __init__(self, cluster_features, kmeans, column_name="Cluster"):
        super().__init__()
        self.cluster_features = cluster_features
        self.kmeans = kmeans
        self.column_name = column_name
        self.actual_cluster_column_names = []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()  # avoid changes to original dataset

        self.actual_cluster_column_names = alike_matches(strings=X_.columns, substrings=self.cluster_features)
        #print(self.actual_cluster_column_names)

        X_cluster = X_.drop(self.actual_cluster_column_names, axis=1)

        X_[self.column_name] = self.kmeans.fit_predict(X_cluster).astype(np.float64)  # It's clustering time!
        #X_[self.column_name] = X_[self.column_name].astype("category")

        #print(X_.dtypes)

        return X_


# use this in pipelines
def add_onehot_cluster(cluster_features, kmeans, column_name="Cluster"):
    onehot_encoder = ColumnTransformer([
        ("onehot encoder", OneHotEncoder(sparse=True, dtype=np.float64), [column_name])
    ], remainder="passthrough")

    return Pipeline([
        ("add kmeans cluster", AddCluster(cluster_features, kmeans, column_name=column_name)),
        ("onehot encode cluster", onehot_encoder)#,
        #("wrap as pandas dataframe", WrapAsDataFrame(prev_preprocessor=onehot_encoder)),
        #("as numpy", AsNumPy(verbose=True))
    ])