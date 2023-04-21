from typing import Tuple, Any
import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd

from yaml import SafeLoader

from src.exception import CustomException


def load_artifact(path: str) -> Any:
    """
    Reads in an artifact as a Python object

    Args:
        path: Artifact file path
    """
    try:
        if path.split(".")[-1] == "yml":
            return yaml.load(open(path), Loader=SafeLoader)
        if path.split(".")[-1] == "pkl":
            return pickle.load(open(path, "rb"))
        if path.split(".")[-1] == "csv":
            return pd.read_csv(path)
    except Exception as err:
        raise CustomException(err, sys) from err


def save_artifact(artifact: Any, path: str):
    """
    Writes artifact to path

    Args:
        artifact: Python object
        path: File path the artifact is written to
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    try:
        if path.split(".")[-1] == "pkl":
            pickle.dump(artifact, open(path, "wb"))
        if path.split(".")[-1] == "csv":
            artifact.to_csv(path, index=False)
    except Exception as err:
        raise CustomException(err, sys) from err


def process_features(
        train_set: pd.DataFrame,
        test_set: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns ML-ready train and test set features and targets

    Args:
        train_set: Train set with missing values
        test_set: Test set with missing values

    Returns:
        X_train: ML-ready train set feature matrix
        X_test: ML-ready test set features matrix
        y_train: Train set target vector
        y_test: Test set target vector
    """
    try:
        # read in './conf/parameters.yml'
        params = load_artifact(r"./conf/parameters.yml")

        # separate the target from the features
        target: str = params["target"]
        x_train, y_train = train_set.drop(target, axis=1), train_set[target]
        x_test, y_test = test_set.drop(target, axis=1), test_set[target]

        # specify the numeric and categorical (nominal and ordinal) features
        numeric_cols: list = params["numeric_features"]
        nominal_cols: list = params["nominal_features"]
        ordinal_cols: list = params["ordinal"]["features"]

        # identify the features that have missing values
        null_cols = [
            col
            for col in numeric_cols + nominal_cols + ordinal_cols
            if x_train[col].isna().sum() > 0
        ]

        # label encode the categorical features
        ctoi, itoc = {}, {}
        for col in nominal_cols + ordinal_cols:
            categories = sorted(set(x_train[col].dropna()))
            indices = range(len(categories))
            ctoi[col] = dict(zip(categories, indices))
            itoc[col] = dict(zip(indices, categories))
            x_train[col] = x_train[col].map(ctoi[col])
            x_test[col] = x_test[col].map(ctoi[col])

        # read in './artifacts/imputer.pkl'
        imputer = load_artifact(r"./artifacts/imputer.pkl")

        # impute the train and test set features
        x_train = pd.DataFrame(
            imputer.transform(x_train),
            columns=x_train.columns.tolist(),
            index=x_train.index.tolist()
        )
        x_test = pd.DataFrame(
            imputer.transform(x_test),
            columns=x_test.columns.tolist(),
            index=x_test.index.tolist()
        )

        # map each categorical feature back to its original categories
        for col in nominal_cols + ordinal_cols:
            if col in null_cols:
                x_train[col] = np.abs(x_train[col]).round().astype(int).map(itoc[col])
                x_test[col] = np.abs(x_test[col]).round().astype(int).map(itoc[col])
            else:
                x_train[col] = x_train[col].astype(int).map(itoc[col])
                x_test[col] = x_test[col].astype(int).map(itoc[col])

        # read in './artifacts/feature_transformer.pkl'
        ft = load_artifact(r"./artifacts/feature_transformer.pkl")

        # extract the 'ft' object's one-hot encoded features
        ohe_cols = []
        for array in ft.transformers_[1][1].categories_:
            ohe_cols += [
                category.lower().replace(" ", "_").replace("/", "_")
                for category in array.tolist()
            ]

        # transform the train and test set features
        x_train = pd.DataFrame(
            ft.transform(x_train),
            columns=numeric_cols + ohe_cols + ordinal_cols,
            index=x_train.index.tolist()
        )
        x_test = pd.DataFrame(
            ft.transform(x_test),
            columns=numeric_cols + ohe_cols + ordinal_cols,
            index=x_test.index.tolist()
        )

        # remove duplicate features
        x_train = x_train.T.drop_duplicates(keep="first").T.copy(deep=True)
        x_test = x_test.T.drop_duplicates(keep="first").T.copy(deep=True)
        return x_train, x_test, y_train, y_test
    except Exception as err:
        raise CustomException(err, sys) from err
