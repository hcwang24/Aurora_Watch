import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta

# Visualization packages
import plotly.express as px

# Evaluation metrics
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

# Sktime forecasting and model selection
from sktime.forecasting.model_selection import SlidingWindowSplitter, ForecastingRandomizedSearchCV
from sktime.utils.plotting import plot_series
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.compose import EnsembleForecaster
from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
from sktime.regression.deep_learning import CNNRegressor, InceptionTimeRegressor


# Machine learning models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Additional forecasting libraries
import pmdarima as pm

# Data preprocessing
from scipy.stats import uniform, randint

class MachineLearningModelTraining:
    """
    A class to handle machine learning model training for time-series forecasting tasks using 
    multiple algorithms, with options for fine-tuning, evaluation, and model prediction.

    Attributes:
    ----------
    df : pd.DataFrame
        The DataFrame containing the time-series data with 'Datetime' as the index.
    save_dir : str
        Directory where the trained models will be saved.
    forecasters : dict
        Dictionary containing forecasters for various models.
    param_grids : dict
        Dictionary containing hyperparameter grids for each model used for fine-tuning.
    results : dict
        Dictionary storing model performance results (MAPE).
    forecasters_results : dict
        Stores model prediction results for validation sets.
    best_forecasters : dict
        Stores the best-tuned forecasters after hyperparameter tuning.

    Methods:
    -------
    split_data(test_start_date, valid_start_date):
        Splits data into train, validation, and test sets based on provided dates.

    train_all_models(X_train, y_train, X_valid, y_valid):
        Trains all models and evaluates their performance using MAPE.

    train_specific_model(model_name, X_train, y_train, X_valid, y_valid):
        Trains a specific model and evaluates its performance.

    fine_tune_model(model_name, X_train, y_train, cv):
        Fine-tunes a specific model using RandomizedSearchCV and cross-validation.

    evaluate(y_train, X_train, cv, model_name=None):
        Evaluates trained models using cross-validation.

    plot_results():
        Plots prediction results for all models.

    get_scores():
        Returns the performance metrics (MAPE) of all models.

    make_predictions(model_name, X_test):
        Makes predictions on the test set using a specified model.
    """
    
    def __init__(self, df, save_dir):
        self.df = df.set_index("Datetime")
        self.save_dir = save_dir
        self.forecasters = {
            "NaiveForecaster": NaiveForecaster(strategy="last"),
            "RandomForest": make_reduction(RandomForestRegressor(n_estimators=100, random_state=123), strategy="recursive", window_length=27),
            "GradientBoosting": make_reduction(GradientBoostingRegressor(n_estimators=100, random_state=123), strategy="recursive", window_length=27),
            "XGBoost": make_reduction(XGBRegressor(n_estimators=100, random_state=123), strategy="recursive", window_length=27),
            "LightGBM": make_reduction(LGBMRegressor(n_estimators=100, random_state=123), strategy="recursive", window_length=27),
            "CatBoost": make_reduction(CatBoostRegressor(n_estimators=100, random_state=123, silent=True), strategy="recursive", window_length=27),
            "CNNRegressor": make_reduction(CNNRegressor(random_state=123, n_epochs=3000), strategy="recursive", window_length=27),
            "InceptionTimeRegressor": make_reduction(InceptionTimeRegressor(random_state=123, n_epochs=3000), strategy="recursive", window_length=27)
        }
        self.param_grids = {
            "LightGBM": {
                "num_leaves": randint(10, 100),
                "max_depth": randint(3, 8),
                "learning_rate": uniform(0.01, 0.30),
            },
            "XGBoost": {
                "max_depth": randint(3, 8),
                "learning_rate": uniform(0.01, 0.30),
            },
            "GradientBoosting": {
                "max_depth": randint(3, 8),
                "learning_rate": uniform(0.01, 0.30),
                "n_estimators": randint(10, 100),
            },
            "RandomForest": {
                "n_estimators": randint(10, 100),
                "max_depth": randint(5, 21),
                "min_samples_split": randint(2, 11),
            },
            "CatBoost": {
                "iterations": randint(10, 100),
                "learning_rate": uniform(0.01, 0.30),
                "depth": randint(3, 10),
            }
        }
        self.results = {
            'Model': [],
            'MAPE': [],
        }
        self.forecasters_results = {}
        self.best_forecasters = {}

    def split_data(self, test_start_date=(datetime.today()-timedelta(days=365)), valid_start_date=(datetime.today()-timedelta(days=365*2))):
        """
        Splits data into train, validation, and test sets based on the given date ranges.
        """
        if test_start_date > valid_start_date:
            print("The dates for test and valid set are not set correctly!")
        else:
            train_set = self.df[self.df.index < valid_start_date]
            valid_set = self.df[(self.df.index >= valid_start_date) & (self.df.index < test_start_date)]
            test_set = self.df[self.df.index >= test_start_date]
            X_train, y_train = train_set.drop(columns=['Kp_Index']), train_set['Kp_Index']
            X_valid, y_valid = valid_set.drop(columns=['Kp_Index']), valid_set['Kp_Index']
            X_test, y_test = test_set.drop(columns=['Kp_Index']), test_set['Kp_Index']
            fh = ForecastingHorizon(np.arange(1,len(y_valid)+1))
            cv = SlidingWindowSplitter(window_length=len(y_test), step_length=int(len(y_test)/2), fh=fh)
            return X_train, y_train, X_valid, y_valid, X_test, y_test, fh, cv

    def train_all_models(self, X_train, y_train, X_valid, y_valid, fh):
        """
        Trains all available models and saves the trained models. 
        Also evaluates them on the validation set using MAPE.
        """
        for name, forecaster in self.forecasters.items():
            print(f"Training {name}...")
            forecaster.fit(y_train, X=X_train, fh=fh)
            prediction = forecaster.predict(X=X_valid, fh=fh)
            self.forecasters_results[name] = prediction
            self.results['Model'].append(name)
            self.results['MAPE'].append(mean_absolute_percentage_error(y_valid, prediction))
            joblib.dump(forecaster, self.save_dir + name + ".pkl")
            print(f"Finished training {name}.")
        return pd.DataFrame(self.results).set_index("Model").sort_values(by="MAPE")

    def train_specific_model(self, model_name, X_train, y_train, X_valid, y_valid, fh):
        """
        Trains a specific model and evaluates it on the validation set using MAPE.
        """
        if model_name not in self.forecasters:
            print(f"Model {model_name} not found!")
            return
        print(f"Training {model_name}...")
        forecaster = self.forecasters[model_name]
        forecaster.fit(y_train, X=X_train, fh=fh)
        prediction = forecaster.predict(X=X_valid, fh=fh)
        self.forecasters_results[model_name] = prediction
        self.results['Model'].append(model_name)
        self.results['MAPE'].append(mean_absolute_percentage_error(y_valid, prediction))
        joblib.dump(forecaster, self.save_dir + model_name + ".pkl")
        print(f"{model_name} has been trained.")
        return pd.DataFrame(self.results).set_index("Model").sort_values(by="MAPE")

    def fine_tune_model(self, model_name, X_train, y_train, cv):
        """
        Fine-tunes a specific model using ForecastingRandomizedSearchCV.
        """
        n_splits = cv.get_n_splits(y_train)
        print(f"Training with {n_splits} fold cross-validation.")
        if model_name not in self.param_grids:
            print(f"No parameter grid for {model_name}.")
            return
        forecaster = self.forecasters[model_name]
        search = ForecastingRandomizedSearchCV(
            forecaster, param_distributions=self.param_grids[model_name], n_iter=20, strategy='refit', random_state=123, cv=cv, n_jobs=-1)
        search.fit(y_train, X=X_train)
        best_forecaster = search.best_forecaster_
        self.best_forecasters[model_name] = best_forecaster
        joblib.dump(best_forecaster, self.save_dir + model_name + "_fine_tuned.pkl")
        print(f"{model_name} has been fine-tuned.")
        return best_forecaster

    def evaluate(self, y_train, X_train, cv, model_name=None):
        """
        Evaluates models using cross-validation. Optionally evaluates only a specific model.
        """
        if model_name:
            forecaster = self.best_forecasters.get(model_name, self.forecasters.get(model_name))
            if not forecaster:
                print(f"Model {model_name} is not trained yet.")
                return
            results = evaluate(forecaster=forecaster, y=y_train, X=X_train, cv=cv, strategy='refit', scoring=mean_absolute_percentage_error)
            print(f"Evaluation result for {model_name}: {results}")
        else:
            for name, forecaster in self.forecasters.items():
                print(f"Evaluating {name}...")
                results = evaluate(forecaster=forecaster, y=y_train, X=X_train, cv=cv, strategy='refit', scoring=mean_absolute_percentage_error)
                print(f"Evaluation result for {name}: {results}")

    def plot_results(self):
        """
        Plots the validation predictions of all models.
        """
        for name, y_pred in self.forecasters_results.items():
            fig = px.line(title=f"{name} Prediction vs Actual")
            fig.add_scatter(x=self.df.index, y=y_pred, name="Predicted")
            fig.add_scatter(x=self.df.index, y=self.df['Kp_Index'], name="Actual")
            fig.show()

    def get_scores(self):
        """
        Returns the performance metrics (MAPE) of all models.
        """
        return pd.DataFrame(self.results).set_index("Model").sort_values(by="MAPE")

    def make_predictions(self, model_name, X_test):
        """
        Makes predictions using a specified model on the test set.
        """
        model = joblib.load(self.save_dir + model_name + "_trained.pkl")
        predictions = model.predict(X=X_test)
        return predictions
