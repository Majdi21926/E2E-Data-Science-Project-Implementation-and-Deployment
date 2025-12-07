# This is gonna be a script for data transformation
import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation
        """
        try:
            logging.info("Data Transformation initiated")

            # Identify numerical and categorical columns
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical and categorical pipelines created")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            logging.info("ColumnTransformer created")

            return preprocessor
        
        except Exception as e:
            logging.info("Exception occurred in get_data_transformer_object")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        This function is responsible for transforming the data
        """
        try:
            logging.info("Entered the data transformation method or component")

            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            # Seperate features and target
            target_column_name = "math_score"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Separated features and target columns")

            # Create preprocessor with identified columns
            preprocessor = self.get_data_transformer_object()
            logging.info("Preprocessor object created")

            # Fit and transform train data
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Applied preprocessing object on training dataframe")

            # Transform test data
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Applied preprocessing object on test dataframe")

            # Create arrays combining transformed features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Combined transformed features with target")

            # Save preprocessor object
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)

            with open(self.data_transformation_config.preprocessor_obj_file_path, "wb") as file_obj:
                pickle.dump(preprocessor, file_obj)
            logging.info("Saved preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occurred in initiate_data_transformation")
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    # First run data ingestion
    objdi = DataIngestion()
    train_data_path, test_data_path, _ = objdi.initiate_data_ingestion()

    # Then run data transformation
    objdt = DataTransformation()
    train_arr, test_arr, preprocessor_path = objdt.initiate_data_transformation(train_data_path, test_data_path)
    logging.info("Data transformation completed successfully")




