import os
import sys
from dataclasses import dataclass

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('../../artifacts', 'model', 'model.pkl')


class ModelTrainner:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split Training and Test input data.')
            x_train, y_train, x_test, y_test = (train_array[:, :-1],
                                                train_array[:, -1],
                                                test_array[:, :-1],
                                                test_array[:, -1])

            models = {'Random Forest': RandomForestRegressor(),
                      'Decision Tree': DecisionTreeRegressor(),
                      'Gradient Boosting': GradientBoostingRegressor(),
                      'Linear Regression': LinearRegression(),
                      'XGBoost Regression': XGBRegressor(),
                      'Catboost Regression': CatBoostRegressor(verbose=False),
                      'Adaboost Regression': AdaBoostRegressor()}

            params = {"Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},
                      'Decision Tree': {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
                      'Gradient Boosting': {'learning_rate': [0.1, 0.01, 0.05, 0.001],
                                            'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                                            'n_estimators': [8, 16, 32, 64, 128, 256]},
                      'Linear Regression': {},
                      'XGBoost Regression': {'learning_rate': [0.1, 0.01, 0.05, 0.001],
                                             'n_estimators': [8, 16, 32, 64, 128, 256]},
                      'Catboost Regression': {'depth': [6, 8, 10],
                                              'learning_rate': [0.01, 0.05, 0.1],
                                              'iterations': [30, 50, 100]},
                      'Adaboost Regression': {'learning_rate': [0.1, 0.01, 0.5, 0.001],
                                              'n_estimators': [8, 16, 32, 64, 128, 256]}
                      }

            model_report: dict = evaluate_models(x_train=x_train,
                                                 y_train=y_train,
                                                 x_test=x_test,
                                                 y_test=y_test,
                                                 models_to_evaluate=models,
                                                 params=params)
            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found')
            logging.info('Best Model found on both training and testing dataset.')

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)

            y_predicted = best_model.predict(x_test)
            r2_score_pred = r2_score(y_true=y_test, y_pred=y_predicted)
            return r2_score_pred

        except Exception as ex:
            CustomException(ex, sys)
