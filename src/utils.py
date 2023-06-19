import os
import sys
import joblib
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exception import CustomException


def data_loader(file_path: str):
    try:
        return pd.read_csv(filepath_or_buffer=file_path)
    except Exception as ex:
        CustomException(ex, sys)


def save_dataframe(dataframe, path, index=False, header=True):
    try:
        dataframe.to_csv(path, index=index, header=header)
    except Exception as ex:
        CustomException(ex, sys)


def save_object(save_path, obj):
    try:
        dir_path = os.path.dirname(save_path)
        os.makedirs(name=dir_path, exist_ok=True)
        joblib.dump(obj, save_path)
    except Exception as ex:
        raise CustomException(ex, sys)


def load_object(object_path):
    try:
        return joblib.load(object_path)
    except Exception as ex:
        raise CustomException(ex, sys)


def evaluate_models(x_train, y_train, x_test, y_test, models_to_evaluate, params, cv=3, n_jobs=-1):
    try:
        report = {}
        for i in range(len(list(models_to_evaluate))):
            model_name = list(models_to_evaluate.keys())[i]
            model = models_to_evaluate[model_name]
            param = params[model_name]

            grid_search = GridSearchCV(estimator=model,
                                       param_grid=param,
                                       cv=cv,
                                       n_jobs=n_jobs)
            grid_search.fit(x_train, y_train)

            model.set_params(**grid_search.best_params_)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            test_model_score = r2_score(y_test, y_pred)
            report[list(models_to_evaluate.keys())[i]] = test_model_score
            logging.info(f'Tested data for model : {model_name}.')
        return report
    except Exception as ex:
        CustomException(ex, sys)
