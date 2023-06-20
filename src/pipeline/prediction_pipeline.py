import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.utils import load_object
from src.exception import CustomException


@dataclass
class PredictionPipelineConfig:
    model_path = os.path.join('artifacts', 'model', 'model.sav')
    preprocessor_path = os.path.join('artifacts', 'preprocessor', 'preprocessor.sav')


class PredictionPipeline:
    def __init__(self):
        self.prediction_pipeline_config = PredictionPipelineConfig()

    def predict(self, features):
        try:
            model = load_object(object_path=self.prediction_pipeline_config.model_path)
            preprocessor = load_object(object_path=self.prediction_pipeline_config.preprocessor_path)
            scaled_data = preprocessor.transform(features)
            return model.predict(scaled_data)
        except Exception as ex:
            CustomException(ex, sys)


class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 writing_score: int,
                 reading_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.writing_score = writing_score
        self.reading_score = reading_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'gender': [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course],
                'writing_score': [self.writing_score],
                'reading_score': [self.reading_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            CustomException(e, sys)
