import sys
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import load_artifact, get_adj_rsquared
from src.components.ingest import DataIngestion
from src.components.transform import DataTransformation


class ModelEvaluation:
    def __init__(self, features: pd.DataFrame, target: pd.Series):
        self.X: pd.DataFrame = features
        self.y: pd.Series = target

    def evaluate(self):
        try:
            model = load_artifact(r"./artifacts/model.pkl")
            yhat: np.ndarray = model.predict(self.X)
            metric: float = get_adj_rsquared(self.X, self.y, yhat)
            logging.info(
                "%s, the highest test set adjusted R², was produced via %s",
                np.round(metric, 2),
                str(model).replace("()", "")
            )
        except Exception as err:
            raise CustomException(err, sys) from err


if __name__ == "__main__":
    train_data_path, test_data_path = DataIngestion().ingest()
    _, X_test, _, y_test = DataTransformation(train_data_path, test_data_path).transform()
    ModelEvaluation(X_test, y_test).evaluate()
