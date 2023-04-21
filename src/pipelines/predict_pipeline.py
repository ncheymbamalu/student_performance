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
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

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
    def __init__(self):
        self.ft = load_artifact(r"./artifacts/feature_transformer.pkl")
        self.model = load_artifact(r"./artifacts/model.pkl")

    def predict(self, record: pd.DataFrame):
        try:
            numeric_features: list = self.ft.transformers_[0][2]
            ordinal_features: list = self.ft.transformers_[2][2]
            ohe_features = []
            for array in self.ft.transformers_[1][1].categories_:
                ohe_features += [
                    category.lower().replace(" ", "_").replace("/", "_")
                    for category in array.tolist()
                ]
            record = pd.DataFrame(
                self.ft.transform(record),
                columns=numeric_features + ohe_features + ordinal_features,
                index=record.index.tolist()
            )
            prediction = int(self.model.predict(record)[0])
            if prediction > 100:
                prediction = 100
            elif prediction < 0:
                prediction = 0
            return prediction
        except Exception as err:
            raise CustomException(err, sys) from err
