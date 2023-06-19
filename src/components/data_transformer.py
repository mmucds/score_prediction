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
from src.utils import save_object, data_loader
from src.exception import CustomException


@dataclass
class DataTransformerConfig:
    preprocessor_obj_path = os.path.join('../../artifacts',
                                         'preprocessor',
                                         'preprocessor.sav')
    numerical_columns = ['writing_score',
                         'reading_score']
    categorical_columns = ['gender',
                           'race_ethnicity',
                           'parental_level_of_education',
                           'lunch',
                           'test_preparation_course']
    target_column_name = 'math_score'


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
                       ('scaler', StandardScaler())])
            cat_pipeline = Pipeline(
                steps=[('impute', SimpleImputer(strategy='most_frequent')),
                       ('one_hot_encoder', OneHotEncoder()),
                       ('scaler', StandardScaler(with_mean=False))])

            logging.info(f'Categorical columns : {self.data_transformer_config.categorical_columns}')
            logging.info(f'Numerical columns : {self.data_transformer_config.numerical_columns}')
            preprocessor = ColumnTransformer(
                [('num_pipeline', num_pipeline, self.data_transformer_config.numerical_columns),
                 ('categorical_columns', cat_pipeline, self.data_transformer_config.categorical_columns)
                 ])
            return preprocessor
        except Exception as ex:
            raise CustomException(ex, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = data_loader(file_path=train_data_path)
            logging.info('Train data read completed.')

            test_df = pd.read_csv(test_data_path)
            logging.info('Test data read completed.')

            logging.info('Obtaining preprocessor object.')
            preprocessing_obj = self.create_data_transformer_object()

            input_feature_train_df = train_df.drop(columns=[self.data_transformer_config.target_column_name],
                                                   axis=1)
            target_feature_train_df = train_df[self.data_transformer_config.target_column_name]

            input_feature_test_df = test_df.drop(columns=[self.data_transformer_config.target_column_name],
                                                 axis=1)
            target_feature_test_df = test_df[self.data_transformer_config.target_column_name]

            logging.info('Applying preprocessing object on training dataframe and testing dataframe.')
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('Saving preprocessing object.')
            save_object(save_path=self.data_transformer_config.preprocessor_obj_path,
                        obj=preprocessing_obj)
            return train_arr, test_arr, self.data_transformer_config.preprocessor_obj_path

        except Exception as ex:
            raise CustomException(ex, sys)
