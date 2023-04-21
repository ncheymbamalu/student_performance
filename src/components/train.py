import sys
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import load_artifact, get_adj_rsquared
from src.components.ingest import DataIngestion
from src.components.transform import DataTransformation


class Train:
    def __init__(self):
        self.model = load_artifact(r"./artifacts/model.pkl")

    def evaluate(self, feature_matrix: pd.DataFrame, target_vector: pd.Series):
        try:
            prediction_vector: np.ndarray = self.model.predict(feature_matrix)
            metric = get_adj_rsquared(
                feature_matrix,
                target_vector,
                prediction_vector
            )
            logging.info(
                "%s, the highest test set adjusted R², was produced via %s",
                np.round(metric, 2),
                str(self.model).replace("()", "")
            )
        except Exception as err:
            raise CustomException(err, sys) from err


if __name__ == "__main__":
    train_data_path, test_data_path = DataIngestion().ingest_data()
    _, x_test, _, y_test = DataTransformation(train_data_path, test_data_path).transform_data()
    Train().evaluate(x_test, y_test)
