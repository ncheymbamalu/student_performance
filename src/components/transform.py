import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import load_artifact, process_features
from src.components.ingest import DataIngestion


class DataTransformation:
    def __init__(self, train_data_path: str, test_data_path: str):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

    def transform_data(self):
        try:
            logging.info("Feature transformation initiated")
            df_train = load_artifact(self.train_data_path)
            df_test = load_artifact(self.test_data_path)
            x_train, x_test, y_train, y_test = process_features(df_train, df_test)
            logging.info("Feature transformation completed")
            return x_train, x_test, y_train, y_test
        except Exception as err:
            raise CustomException(err, sys) from err


if __name__ == "__main__":
    TRAIN_DATA_PATH, TEST_DATA_PATH = DataIngestion().ingest_data()
    _, _, _, _ = DataTransformation(TRAIN_DATA_PATH, TEST_DATA_PATH).transform_data()
