from typing import List

import streamlit as st
import pandas as pd

from src.utils import load_artifact
from src.pipelines.predict_pipeline import CustomData, PredictPipeline

data: pd.DataFrame = load_artifact(r"./artifacts/test.csv")

st.title("Student Performance")

if st.checkbox("Show Dataset"):
    st.dataframe(data)

st.write("### Required Information:")

# categorical features
genders = ["male", "female"]
gender = st.selectbox("Gender", genders)

race_ethnicities: List[str] = sorted(set(data["race_ethnicity"].dropna()))
race_ethnicity = st.selectbox("Race Ethnicity", race_ethnicities)

parent_education_levels = [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree"
]
parental_level_of_education = st.selectbox("Parents' Education Level", parent_education_levels)

lunch_types: List[str] = sorted(set(data["lunch"].dropna()))
lunch = st.selectbox("Lunch Type", lunch_types)

test_prep_choices: List[str] = sorted(set(data["test_preparation_course"].dropna()))
test_preparation_course = st.selectbox("Test Preparation Course", test_prep_choices)

# numeric features
reading_score = st.slider(
    "Reading Score",
    0,
    int(data["reading_score"].max()),
    20
)

writing_score = st.slider(
    "Writing Score",
    0,
    int(data["writing_score"].max()),
    20
)

button = st.button("Predict Math Score")

# prediction
if button:
    record: pd.DataFrame = (
        CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )
        .to_dataframe()
    )
    prediction: int = PredictPipeline(record).predict()
    st.subheader(f"The estimated math score is {prediction} (out of 100)")
