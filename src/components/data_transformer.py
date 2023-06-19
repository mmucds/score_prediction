import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from src.logger import logging
from src.utils import save_object
from src.exception import CustomException


@dataclass
class DataTransformerConfig:
    preprocessor_obj_path = os.path.join('artifacts',
                                         'preprocessor',
                                         'preprocessor.sav')
    numerical_columns = ['writing_score',
                         'reading_score']
    categorical_columns = ['gender',
                           'race_ethnicity',
                           'parental_level_of_education',
                           'lunch',
                           'test_preparation_course']


class DataTransformer:
    def __init__(self):
        self.data_transformer_config = DataTransformerConfig()

    def create_data_transformer_object(self):
        """
        This function is creates data transformation pipeline.
        """
        try:
            num_pipeline = Pipeline(
                steps=[('impute', SimpleImputer(strategy='median')),
                       ('scaler', StandardScaler())
                       ])
            cat_pipeline = Pipeline(
                steps=[('impute', SimpleImputer(strategy='most_frequent')),
                       ('one_hot_encoder', OneHotEncoder()),
                       ('scaler', StandardScaler(with_mean=False))
                       ])
            logging.info(f'Categorical columns : {self.categorical_columns}')
            logging.info(f'Numerical columns : {self.numerical_columns}')
            preprocessor = ColumnTransformer(
                [('num_pipeline', num_pipeline, self.numerical_columns),
                 ('categorical_columns', cat_pipeline, self.categorical_columns)
                 ])
            return preprocessor
        except Exception as ex:
            raise CustomException(ex, sys)


