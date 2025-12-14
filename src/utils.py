import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging
import sys
from src.exception import CustomException
import os
import pickle

def evaluate_models(X_train, y_train, X_test, y_test, models, params=None):
    """
    This function evaluates multiple regression models with optional hyperparameter tuning
    using GridSearchCV and returns a report with performance metrics.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models: Dictionary of model names and model instances
        params: Dictionary of hyperparameter grids for each model (optional)
                If None or empty for a model, no tuning is performed.
    
    Returns:
        dict: Dictionary containing model performance metrics and best model information
    """
    try:
        report = {}
        
        for model_name, model in models.items():
            logging.info(f"Training {model_name}...")
            
            # Check if hyperparameters are provided for this model
            model_params = params.get(model_name, {}) if params else {}
            
            if model_params:
                # Perform hyperparameter tuning with GridSearchCV
                logging.info(f"Performing hyperparameter tuning for {model_name}...")
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=model_params,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train, y_train)
                
                # Get best model from GridSearchCV
                best_estimator = grid_search.best_estimator_
                logging.info(f"Best params for {model_name}: {grid_search.best_params_}")
            else:
                # Train without hyperparameter tuning
                model.fit(X_train, y_train)
                best_estimator = model
            
            # Make predictions using the best estimator
            y_train_pred = best_estimator.predict(X_train)
            y_test_pred = best_estimator.predict(X_test)
            
            # Evaluate on training set
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_rmse = np.sqrt(train_mse)
            train_r2 = r2_score(y_train, y_train_pred)
            
            # Evaluate on test set
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Store metrics in report
            report[model_name] = {
                'train_mae': train_mae,
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'model': best_estimator  # Store the best trained model
            }
            
            logging.info(f"{model_name} - Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")
        
        # Find the best model based on test R2 score
        best_model_name = max(report.keys(), key=lambda x: report[x]['test_r2'])
        best_model_score = report[best_model_name]['test_r2']
        best_model = report[best_model_name]['model']
        
        logging.info(f"Best model: {best_model_name} with Test R2 Score: {best_model_score:.4f}")
        
        # Add best model info to report
        report['best_model'] = {
            'name': best_model_name,
            'score': best_model_score,
            'model': best_model
        }
        
        return report
        
    except Exception as e:
        logging.info("Exception occurred in evaluate_models")
        raise CustomException(e, sys)


def save_object(file_path: str, obj) -> None:
    """
    Persist a Python object to disk using pickle. Creates parent directories if needed.

    Args:
        file_path (str): Destination file path (will be opened in binary write mode).
        obj: Any picklable Python object to persist.

    Raises:
        CustomException: Wraps any exception raised during saving.
    """
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info(f"Saved object to: {file_path}")
    except Exception as e:
        logging.info("Exception occurred in save_object")
        raise CustomException(e, sys)
    

def load_object(file_path: str):

    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        logging.info(f"Loaded object from: {file_path}")
        return obj
    except Exception as e:
        logging.info("Exception occurred in load_object")
        raise CustomException(e, sys)

