from typing import Any
import os
import sys
import yaml
import pickle
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
        elif path.split(".")[-1] == "pkl":
            return pickle.load(open(path, "rb"))
        elif path.split(".")[-1] == "csv":
            return pd.read_csv(path)
    except Exception as err:
        raise CustomException(err, sys)


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
        elif path.split(".")[-1] == "csv":
            artifact.to_csv(path, index=False)
    except Exception as err:
        raise CustomException(err, sys)
