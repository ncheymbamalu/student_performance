from typing import Dict, Any
import os
import sys

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.utils import load_artifact, save_artifact


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("artifacts", "train_data.csv")
    test_data_path: str = os.path.join("artifacts", "test_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.params: Dict[str, Any] = load_artifact(r"./conf/parameters.yml")

    def ingest_data(self):
        try:
            logging.info("Data ingestion initiated")
            df_raw = load_artifact(r"./data/student_performance.csv")

            logging.info("Splitting data into train and test sets")
            df_train, df_test = train_test_split(
                df_raw,
                test_size=self.params["test_size"],
                random_state=self.params["random_state"]
            )

            logging.info(
                "Saving the raw, train, and test data to %s",
                os.path.join(os.getcwd(), os.path.dirname(self.ingestion_config.raw_data_path))
            )
            save_artifact(df_raw, self.ingestion_config.raw_data_path)
            save_artifact(df_train, self.ingestion_config.train_data_path)
            save_artifact(df_test, self.ingestion_config.test_data_path)
            logging.info("Data ingestion complete")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as err:
            raise CustomException(err, sys) from err


if __name__ == "__main__":
    _, _ = DataIngestion().ingest_data()
