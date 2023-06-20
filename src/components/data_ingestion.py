import os
import sys

import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.utils import data_loader, save_dataframe
from src.components.data_transformer import DataTransformer
from src.components.model_trainer import ModelTrainner



@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('../../data', 'raw', 'StudentsPerformance.csv')
    train_data_path: str = os.path.join('../../artifacts', 'data', 'train.csv')
    test_data_path: str = os.path.join('../../artifacts', 'data', 'test.csv')


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component.')
        try:
            df = data_loader(file_path=self.data_ingestion_config.raw_data_path)
            logging.info('Read the dataset as dataframe.')
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),
                        exist_ok=True)

            logging.info('Train Test split initiated.')
            train_df, test_df = train_test_split(df,
                                                 test_size=0.3,
                                                 random_state=101)

            save_dataframe(dataframe=train_df,
                           path=self.data_ingestion_config.train_data_path)
            save_dataframe(dataframe=test_df,
                           path=self.data_ingestion_config.test_data_path)

            logging.info('Data ingestion is completed.')

            return self.data_ingestion_config.train_data_path, \
                self.data_ingestion_config.test_data_path

        except Exception as ex:
            raise CustomException(ex, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_df, test_df = obj.initiate_data_ingestion()

    data_transformer = DataTransformer()
    train_arr, test_arr, _ = data_transformer.initiate_data_transformation(train_df, test_df)

    model_trainer_obj = ModelTrainner()
    print(model_trainer_obj.initiate_model_trainer(train_arr, test_arr))
