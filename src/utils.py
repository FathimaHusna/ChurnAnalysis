import os
import sys
import dill
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.combine import SMOTEENN
from sklearn.model_selection import RandomizedSearchCV
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Save a Python object to a file using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param_grids=None, n_iter=10, cv=3):
    """
    Evaluate multiple models with optional hyperparameter tuning.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param X_test: Test features
    :param y_test: Test labels
    :param models: Dictionary of model names and instances
    :param param_grids: Dictionary of model names and hyperparameter grids for tuning
    :param n_iter: Number of iterations for RandomizedSearchCV
    :param cv: Number of cross-validation folds
    :return: Dictionary of model names and their accuracy scores
    """
    model_report = {}

    for model_name, model in models.items():
        if param_grids and model_name in param_grids:
            try:
                # Perform hyperparameter tuning
                search = RandomizedSearchCV(
                    model, param_grids[model_name], n_iter=n_iter, cv=cv, verbose=2, n_jobs=-1
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                logging.info(f"Best hyperparameters for {model_name}: {search.best_params_}")
            except Exception as e:
                raise CustomException(e, sys)
        else:
            best_model = model
            best_model.fit(X_train, y_train)

        try:
            # Predict and evaluate the model
            y_pred = best_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred)
            model_report[model_name] = acc

            print(f"Model: {model_name}")
            print(f"Accuracy: {acc}")
            print("Confusion Matrix:")
            print(cm)
            print("Classification Report:")
            print(cr)
            print("="*60)
        except Exception as e:
            raise CustomException(e, sys)

    return model_report

def balance_data(X, y):
    """
    Balance the dataset using SMOTEENN to handle imbalanced data.
    
    :param X: Features
    :param y: Labels
    :return: Resampled features and labels
    """
    try:
        smote_enn = SMOTEENN()
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        return X_resampled, y_resampled
    except Exception as e:
        raise CustomException(e, sys)
