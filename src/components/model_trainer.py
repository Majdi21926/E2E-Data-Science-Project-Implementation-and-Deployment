# Last, we implemented data_ingestion and data_transformation components.
# This is goona be a script for model training
import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion
from src.utils import evaluate_models, save_object
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

import pickle
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1]
            )
            logging.info("Splitting training and testing input data Completed")

            models = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest Regressor": RandomForestRegressor(),
                    "XGBRegressor": XGBRegressor(),
                    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                    "AdaBoost Regressor": AdaBoostRegressor()
            }

            best_model_report:dict = evaluate_models(X_train, y_train, X_test, y_test, models)
            best_model_score = best_model_report['best_model']['score']

            if best_model_score < 0.6:
                raise CustomException('No best model found')
            
            logging.info("Best model found on both training and testing dataset")

            best_model_name = best_model_report['best_model']['name']
            best_model = models[best_model_name]

            if best_model_name == 'Linear Regression':
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )
            else:
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )
            
            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square
            
        except Exception as e:
            logging.info("Exception occurred in initiate_model_training")
            raise CustomException(e, sys)
        

if __name__ == '__main__':
    objmt = ModelTrainer()
    train_data, test_data, _ = DataIngestion().initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    r2_score = objmt.initiate_model_training(train_arr, test_arr)
    print(r2_score)
    logging.info(f"R2 Score: {r2_score}")
    logging.info("Model Training Completed")
