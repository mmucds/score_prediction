import os
import sys
import joblib

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exception import CustomException


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
