from typing import Dict, Any
import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import load_artifact, process_features
from src.components.ingestion import DataIngestion


class DataTransformation:
    def __init__(self):
        self.params: Dict[str, Any] = load_artifact(r"./conf/parameters.yml")

    def transform_data(self, train_data_path: str, test_data_path: str):
        try:
            logging.info("Feature transformation initiated")
            df_train = load_artifact(train_data_path)
            df_test = load_artifact(test_data_path)
            x_train, x_test, y_train, y_test = process_features(df_train, df_test)
            logging.info("Feature transformation completed")
            return x_train, x_test, y_train, y_test
        except Exception as err:
            raise CustomException(err, sys) from err


if __name__ == "__main__":
    TRAIN_DATA_PATH, TEST_DATA_PATH = DataIngestion().ingest_data()
    _, _, _, _ = DataTransformation().transform_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
