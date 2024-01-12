from typing import Tuple, List, Dict, Union, Any
from pathlib import Path
import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd

from yaml import SafeLoader

from src.exception import CustomException


def load_artifact(filepath: str) -> Any:
    """
    Reads in filepath as a Python object

    Args:
        filepath: Artifact file path
    """
    try:
        if Path(filepath).suffix == ".yml":
            return yaml.load(open(filepath), Loader=SafeLoader)
        if Path(filepath).suffix == ".pkl":
            return pickle.load(open(filepath, "rb"))
        if Path(filepath).suffix == ".csv":
            return pd.read_csv(filepath)
    except Exception as err:
        raise CustomException(err, sys) from err


def save_artifact(python_object: Any, filepath: str):
    """
    Writes python_object to filepath

    Args:
        python_object: Python object
        filepath: File path where python_object is written to
    """
    try:
        directory: str = os.path.dirname(filepath)
        os.makedirs(directory, exist_ok=True)
        if Path(filepath).suffix == ".pkl":
            pickle.dump(python_object, open(filepath, "wb"))
        if Path(filepath).suffix == ".csv":
            python_object.to_csv(filepath, index=False)
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
        x_train: ML-ready train set feature matrix
        x_test: ML-ready test set features matrix
        y_train: Train set target vector
        y_test: Test set target vector
    """
    try:
        # read in './conf/parameters.yml'
        params: Dict[str, Any] = load_artifact(r"./conf/parameters.yml")

        # separate the features and target
        target: str = params.get("target")
        x_train: pd.DataFrame = train_set.drop(target, axis=1)
        y_train: pd.Series = train_set[target]
        x_test: pd.DataFrame = test_set.drop(target, axis=1)
        y_test: pd.Series = test_set[target]

        # specify the numeric and categorical (nominal and ordinal) features
        numeric_cols: List[str] = params.get("numeric_features")
        nominal_cols: List[str] = params.get("nominal_features")
        ordinal_cols: List[str] = params.get("ordinal").get("features")

        # identify the features that have missing values
        null_cols: List[str] = [
            col
            for col in numeric_cols + nominal_cols + ordinal_cols
            if x_train[col].isna().sum() > 0
        ]

        # label encode the categorical features
        itoc: Dict[str, Dict[int, str]] = {}
        for col in nominal_cols + ordinal_cols:
            categories: List[str] = sorted(set(x_train[col].dropna()))
            indices: range = range(len(categories))
            itoc[col] = dict(zip(indices, categories))
            x_train[col] = x_train[col].map(dict(zip(categories, indices)))
            x_test[col] = x_test[col].map(dict(zip(categories, indices)))

        # read in './artifacts/imputer.pkl'
        imputer = load_artifact(r"./artifacts/imputer.pkl")

        # impute the train and test set features
        x_train = pd.DataFrame(
            imputer.transform(x_train),
            index=x_train.index.tolist(),
            columns=x_train.columns.tolist()
        )
        x_test = pd.DataFrame(
            imputer.transform(x_test),
            index=x_test.index.tolist(),
            columns=x_test.columns.tolist()
        )

        # map each categorical feature back to its original categories
        for col in nominal_cols + ordinal_cols:
            if col in null_cols:
                x_train[col] = np.abs(x_train[col]).round().astype(int).map(itoc.get(col))
                x_test[col] = np.abs(x_test[col]).round().astype(int).map(itoc.get(col))
            else:
                x_train[col] = x_train[col].astype(int).map(itoc.get(col))
                x_test[col] = x_test[col].astype(int).map(itoc.get(col))

        # read in './artifacts/feature_transformer.pkl'
        ft = load_artifact(r"./artifacts/feature_transformer.pkl")

        # extract the one-hot encoded features from the 'ft' object
        ohe_cols: List[str] = []
        for array in ft.transformers_[1][1].categories_:
            ohe_cols += [
                category.lower().replace(" ", "_").replace("/", "_")
                for category in array.tolist()
            ]

        # transform the train and test set features
        x_train = pd.DataFrame(
            ft.transform(x_train),
            index=x_train.index.tolist(),
            columns=numeric_cols + ohe_cols + ordinal_cols
        )
        x_test = pd.DataFrame(
            ft.transform(x_test),
            index=x_test.index.tolist(),
            columns=numeric_cols + ohe_cols + ordinal_cols
        )

        # remove duplicate features
        x_train = x_train.T.drop_duplicates(keep="first").T.copy(deep=True)
        x_test = x_test.T.drop_duplicates(keep="first").T.copy(deep=True)
        return x_train, x_test, y_train, y_test
    except Exception as err:
        raise CustomException(err, sys) from err


def get_adj_rsquared(
    feature_matrix: pd.DataFrame,
    target_vector: pd.Series,
    prediction_vector: Union[np.ndarray, pd.Series]
) -> float:
    """
    Returns the adjusted R²

    Args:
        feature_matrix: feature matrix
        target_vector: target vector
        prediction_vector: prediction vector

    Returns:
        adj_rsquared: Adjusted R²
      """
    try:
        n_records, n_features = feature_matrix.shape
        total: pd.Series = target_vector - np.mean(target_vector)
        sum_squared_total: float = total.dot(total)
        error: pd.Series = target_vector - prediction_vector
        sum_squared_error: float = error.dot(error)
        rsquared: float = 1 - (sum_squared_error / sum_squared_total)
        adj_rsquared: float = 1 - (((1 - rsquared)*(n_records - 1))/(n_records - n_features - 1))
        return adj_rsquared
    except Exception as err:
        raise CustomException(err, sys) from err
