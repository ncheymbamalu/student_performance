from typing import List
import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_artifact


class CustomData:
    def __init__(
            self,
            gender: str,
            race_ethnicity: str,
            parental_level_of_education: str,
            lunch: str,
            test_preparation_course: str,
            reading_score: int,
            writing_score: int
    ):
        self.gender: str = gender
        self.race_ethnicity: str = race_ethnicity
        self.parental_level_of_education: str = parental_level_of_education
        self.lunch: str = lunch
        self.test_preparation_course: str = test_preparation_course
        self.reading_score: int = reading_score
        self.writing_score: int = writing_score

    def to_dataframe(self):
        try:
            return pd.DataFrame(
                {
                    "gender": [self.gender],
                    "race_ethnicity": [self.race_ethnicity],
                    "parental_level_of_education": [self.parental_level_of_education],
                    "lunch": [self.lunch],
                    "test_preparation_course": [self.test_preparation_course],
                    "reading_score": [self.reading_score],
                    "writing_score": [self.writing_score]
                }
            )
        except Exception as err:
            raise CustomException(err, sys) from err


class PredictPipeline:
    def __init__(self, record: pd.DataFrame):
        # 'record' is DataFrame of shape (1, D), where D is the number of features
        self.record: pd.DataFrame = record

    def predict(self):
        try:
            ft = load_artifact(r"./artifacts/feature_transformer.pkl")
            model = load_artifact(r"./artifacts/model.pkl")
            numeric_features: List[str] = ft.transformers_[0][2]
            ordinal_features: List[str] = ft.transformers_[2][2]
            ohe_features: List[str] = []
            for array in ft.transformers_[1][1].categories_:
                ohe_features += [
                    category.lower().replace(" ", "_").replace("/", "_")
                    for category in array.tolist()
                ]
            record: pd.DataFrame = pd.DataFrame(
                ft.transform(self.record),
                index=self.record.index,
                columns=numeric_features + ohe_features + ordinal_features
            )
            prediction: int = int(round(model.predict(record)[0]))
            prediction = 100 if prediction > 100 else 0 if prediction < 0 else prediction
            return prediction
        except Exception as err:
            raise CustomException(err, sys) from err
