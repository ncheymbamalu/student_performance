from typing import Dict, Any
from dataclasses import dataclass

import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.utils import load_artifact, save_artifact


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config: DataIngestionConfig = DataIngestionConfig()

    def ingest(self):
        try:
            train_data_path: str = self.ingestion_config.train_data_path
            test_data_path: str = self.ingestion_config.test_data_path
            logging.info("Data ingestion initiated")
            params: Dict[str, Any] = load_artifact(r"./conf/parameters.yml")
            df_raw: pd.DataFrame = load_artifact(r"./data/student_performance.csv")

            logging.info("Splitting data into train and test sets")
            df_train, df_test = train_test_split(df_raw, test_size=params.get("test_size"))

            logging.info(
                "Saving %s and %s to %s",
                os.path.basename(train_data_path),
                os.path.basename(test_data_path),
                f"./{os.path.dirname(train_data_path)}"
            )
            for df, path in zip([df_train, df_test], [train_data_path, test_data_path]):
                save_artifact(df, path)
            logging.info("Data ingestion completed")
            return train_data_path, test_data_path
        except Exception as err:
            raise CustomException(err, sys) from err


if __name__ == "__main__":
    DataIngestion().ingest()
