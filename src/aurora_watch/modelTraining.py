import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
from preprocessing import preprocess_kp_index

# Visualization packages
import plotly.express as px

# Evaluation metrics
from sklearn.metrics import mean_absolute_percentage_error

# Sktime forecasting and model selection
from sktime.forecasting.model_selection import SlidingWindowSplitter, ForecastingRandomizedSearchCV
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.compose import EnsembleForecaster
from sktime.regression.deep_learning import CNNRegressor, InceptionTimeRegressor


# Machine learning models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Data preprocessing
from scipy.stats import uniform, randint
from itertools import product

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
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.forecasters = {
            "NaiveForecaster": NaiveForecaster(strategy="last"),
            "RandomForest": make_reduction(RandomForestRegressor(n_estimators=100, random_state=123), strategy="recursive", window_length=27),
            "GradientBoosting": make_reduction(GradientBoostingRegressor(n_estimators=100, random_state=123), strategy="recursive", window_length=27),
            "XGBoost": make_reduction(XGBRegressor(n_estimators=100, random_state=123), strategy="recursive", window_length=27),
            "LightGBM": make_reduction(LGBMRegressor(n_estimators=100, random_state=123), strategy="recursive", window_length=27),
            "CatBoost": make_reduction(CatBoostRegressor(n_estimators=100, random_state=123, silent=True), strategy="recursive", window_length=27),
            # "CNNRegressor": make_reduction(CNNRegressor(random_state=123, n_epochs=3000), strategy="recursive", window_length=27),
            # "InceptionTimeRegressor": make_reduction(InceptionTimeRegressor(random_state=123, n_epochs=3000), strategy="recursive", window_length=27)
        }
        self.param_grids = {
            "LightGBM": {
                "n_estimators": randint(10, 100),
                "num_leaves": randint(10, 100),
                "max_depth": randint(3, 8),
                "learning_rate": uniform(0.01, 0.30),
            },
            "XGBoost": {
                "n_estimators": randint(10, 100),
                "max_depth": randint(3, 8),
                "learning_rate": uniform(0.01, 0.30),
            },
            "GradientBoosting": {
                "n_estimators": randint(10, 100),
                "max_depth": randint(3, 8),
                "learning_rate": uniform(0.01, 0.30)
            },
            "RandomForest": {
                "n_estimators": randint(10, 100),
                "max_depth": randint(5, 21),
                "min_samples_split": randint(2, 11),
            },
            "CatBoost": {
                "estimator__n_estimators": randint(10, 100),
                "estimator__learning_rate": uniform(0.01, 0.30),
                "estimator__depth": randint(3, 10),
            }
        }
        self.results = {
            'Model': [],
            'MAPE': [],
        }
        self.cv_results = {
            'Raw': [],
            'Model': [],
            'MAPE': [],
            'Std': [],
        }
        self.X_train = []
        self.y_train = []
        self.X_valid = []
        self.y_valid = []
        self.X_test = []
        self.y_test = []
        self.fh = []
        self.cv = []
        self.forecasters_results = {}
        self.best_forecasters = {}
        
    def split_data(self, test_start_date=(datetime.today()-timedelta(days=365)), valid_start_date=(datetime.today()-timedelta(days=365*2))):
        """
        Splits data into train, validation, and test sets based on the given date ranges.
        """
        if test_start_date < valid_start_date:
            print("The dates for test and valid set are not set correctly!")
        else:
            train_set = self.df[self.df.index < valid_start_date]
            valid_set = self.df[(self.df.index >= valid_start_date) & (self.df.index < test_start_date)]
            test_set = self.df[self.df.index >= test_start_date]
            
            # Ensure y is a pd.Series with a compatible DatetimeIndex
            self.X_train, self.y_train = train_set.drop(columns=['Kp_Index']), train_set['Kp_Index']
            self.X_valid, self.y_valid = valid_set.drop(columns=['Kp_Index']), valid_set['Kp_Index']
            self.X_test, self.y_test = test_set.drop(columns=['Kp_Index']), test_set['Kp_Index']

            # Check if indices are in the correct format
            self.X_train.index = pd.to_datetime(self.X_train.index)
            self.X_valid.index = pd.to_datetime(self.X_valid.index)
            self.X_test.index = pd.to_datetime(self.X_test.index)
            self.y_train.index = pd.to_datetime(self.y_train.index)
            self.y_valid.index = pd.to_datetime(self.y_valid.index)
            self.y_test.index = pd.to_datetime(self.y_test.index)

            # Set forecasting horizon and cross-validation splitter
            self.fh = ForecastingHorizon(np.arange(1,len(self.y_valid)+1))
            self.cv = SlidingWindowSplitter(window_length=len(self.y_test), step_length=int(len(self.y_test)/2), fh=self.fh)

    def train_all_models(self):
        """
        Trains all available models and saves the trained models. 
        Also evaluates them on the validation set using MAPE.
        """
        for name, forecaster in self.forecasters.items():
            print(f"Training {name}...")
            forecaster.fit(self.y_train, X=self.X_train, fh=self.fh)
            prediction = forecaster.predict(X=self.X_valid, fh=self.fh)
            self.forecasters_results[name] = prediction
            self.results['Model'].append(name)
            self.results['MAPE'].append(mean_absolute_percentage_error(self.y_valid, prediction))
            joblib.dump(forecaster, self.save_dir + name + "_oneshot.pkl")
            print(f"Finished training {name}.")
        return pd.DataFrame(self.results).set_index("Model").sort_values(by="MAPE")

    def train_specific_model(self, model_name):
        """
        Trains a specific model and evaluates it on the validation set using MAPE.
        """
        if model_name not in self.forecasters:
            print(f"Model {model_name} not found!")
            print(f"Please choose one of the following models:")
            print(self.forecasters.keys())
            return
        print(f"Training {model_name} only ..")
        forecaster = self.forecasters[model_name]
        forecaster.fit(self.y_train, X=self.X_train, fh=self.fh)
        prediction = forecaster.predict(X=self.X_valid, fh=self.fh)
        self.forecasters_results[model_name] = prediction
        self.results['Model'].append(model_name)
        self.results['MAPE'].append(mean_absolute_percentage_error(self.y_valid, prediction))
        joblib.dump(forecaster, self.save_dir + model_name + "_oneshot.pkl")
        print(f"Finished training {model_name}.")
        return pd.DataFrame(self.results).set_index("Model").sort_values(by="MAPE")

    def fine_tune_model(self, model_name):
        """
        Fine-tunes a specific model using ForecastingRandomizedSearchCV.
        """
        if model_name not in self.param_grids:
            print(f"No parameter grid for {model_name}. Skipping!")
            return
        n_splits = self.cv.get_n_splits(self.y_train)
        print(f"Training with {n_splits} fold cross-validation.")
        forecaster = self.forecasters[model_name]
        search = ForecastingRandomizedSearchCV(
            forecaster, cv=self.cv, param_distributions=self.param_grids[model_name], n_iter=10, strategy='refit', random_state=123, error_score='raise')
        search.fit(self.y_train, X=self.X_train)
        best_forecaster = search.best_forecaster_
        self.best_forecasters[model_name] = best_forecaster
        prediction = best_forecaster.predict(X=self.X_valid, fh=self.fh)
        self.forecasters_results[model_name+"_finetuned"] = prediction
        self.results['Model'].append(model_name+"_finetuned")
        self.results['MAPE'].append(mean_absolute_percentage_error(self.y_valid, prediction))
        joblib.dump(best_forecaster, self.save_dir + model_name + "_cv_ft.pkl")
        print(f"{model_name} has been fine-tuned.")
        print(f"The fine-tuned parameters are:")
        print(best_forecaster.estimator.get_params)
        return pd.DataFrame(self.results).set_index("Model").sort_values(by="MAPE")

    def evaluate(self, model_name=None):
        """
        Evaluates models using cross-validation. Optionally evaluates only a specific model.
        """
        if model_name:
            forecaster = self.best_forecasters.get(model_name, self.forecasters.get(model_name))
            if not forecaster:
                print(f"Model {model_name} is not trained yet.")
                return
            results = evaluate(forecaster=forecaster, cv=self.cv, y=self.y_train, X=self.X_train, strategy='refit', scoring=mean_absolute_percentage_error)
            self.cv_results['Raw'].append(results)
            self.cv_results['Model'].append(model_name)
            self.cv_results['MAPE'].append(results.iloc[:,0].mean())
            self.cv_results['Std'].append(results.iloc[:,0].std())
            print(pd.DataFrame(self.cv_results).set_index("Model").sort_values(by="MAPE"))
        else:
            for name, forecaster in self.forecasters.items():
                print(f"Evaluating {name}...")
                results = evaluate(forecaster=forecaster, cv=self.cv, y=self.y_train, X=self.X_train, strategy='refit', scoring=mean_absolute_percentage_error)
                self.cv_results['Raw'].append(results)
                self.cv_results['Model'].append(name)
                self.cv_results['MAPE'].append(results.iloc[:,0].mean())
                self.cv_results['Std'].append(results.iloc[:,0].std())
                print(pd.DataFrame(self.cv_results).set_index("Model").sort_values(by="MAPE"))

    def plot_results(self):
        """
        Plots the validation predictions of all models.
        """
        fig = px.line(title="Prediction vs Actual")
        counter = 0
        for model_name, y_pred in self.forecasters_results.items():
            if counter == 0:
                fig.add_scatter(x=y_pred.index, y=self.df['Kp_Index'], name="Actual")
                counter += 1
            fig.add_scatter(x=y_pred.index, y=y_pred, name=model_name)
        
        fig.show()

    def get_scores(self):
        """
        Returns the performance metrics (MAPE) of all models.
        """
        return pd.DataFrame(self.results).set_index("Model").sort_values(by="MAPE")
    
    def load_saved_models(self, save_dir='./models/', modelnum = None):
        """
        Load pre-trained models from a specified directory.
        """
        # Ensure the directory exists
        if not os.path.exists(save_dir):
            print(f"Directory '{save_dir}' does not exist.")
            return
        
        # Find all .pkl files in the directory
        model_files = [f for f in os.listdir(save_dir) if f.endswith('.pkl')]
        
        if not model_files:
            print("No pre-trained models found in the directory.")
            return
        
        print("Available models:")
        for idx, file in enumerate(model_files):
            print(f"{idx + 1}. {file}")

        if modelnum is None:
            print(f"Rerun by entering the model numbers to load (comma-separated, no space), or 'all' to load everything")
            return
        elif modelnum.lower() == 'all':
            selected_files = model_files
        else:
            try:
                indices = [int(x.strip()) - 1 for x in modelnum.split(',')]
                selected_files = [model_files[i] for i in indices]
            except (ValueError, IndexError):
                print("Invalid input. No models loaded.")
                return

        # Load the selected models
        for file in selected_files:
            model_name = file.replace('.pkl', '')
            self.best_forecasters[model_name] = joblib.load(os.path.join(save_dir, file))
            print(f"Loaded model: {model_name}")

        print("Models loaded successfully.")

    def create_ensemble_model(self, model_indices=None, weights=None):
        """
        Creates an ensemble model from selected models in self.best_forecasters.
        If weights are provided, uses them directly. Otherwise, finds the optimal weights
        via ForecastingRandomizedSearchCV.
        
        Parameters:
        - model_indices (list): List of model indices to use in the ensemble.
        - weights (list): List of weights for each model in the ensemble. If None, 
                          ForecastingRandomizedSearchCV will optimize weights.
        """
        # Check for available models
        if not self.best_forecasters:
            print("No fine-tuned models available. Load or fine-tune models before creating an ensemble.")
            return
        
        print("Available models for ensemble:")
        model_keys = list(self.best_forecasters.keys())
        for idx, model_name in enumerate(model_keys):
            print(f"{idx + 1}. {model_name}")
        
        if model_indices is None:
            print(f"Please specify model_indices parameter with the model numbers to include (comma-separated, no space), or 'all' to load everything")
            return
        elif model_indices.lower() == 'all':
            selected_models = model_keys
        else:
            try:
                indices = [int(x.strip()) - 1 for x in model_indices.split(',')]
                selected_models = [model_keys[i] for i in indices]
            except IndexError:
                print("Invalid model indices. No ensemble created.")
                return
        
        # Create the EnsembleForecaster with selected models
        ensemble_forecaster = EnsembleForecaster(forecasters=[(model, self.best_forecasters[model]) for model in selected_models])
        
        # Use provided weights or search for the best weights if not provided
        if weights is not None:
            if len(weights) != len(selected_models) or not np.isclose(sum(weights), 1.0):
                print("Invalid weights: Ensure they sum to 1 and match the number of selected models.")
                return
            ensemble_forecaster.set_params(weights=weights)
            ensemble_forecaster.fit(self.y_train, X=self.X_train)
            print("Ensemble model created with specified weights.")
        else:
            # Set up weight options and search with ForecastingRandomizedSearchCV
            n_models = len(selected_models)
            weight_options = np.arange(0, 1.05, 0.05)
            param_grid = {'weights': [w for w in product(weight_options, repeat=n_models) if sum(w) == 1]}        
            search = ForecastingRandomizedSearchCV(
                forecaster=ensemble_forecaster,
                cv=self.cv,
                param_distributions=param_grid,
                n_iter=10,
                random_state=123,
                n_jobs=-1
            )
            search.fit(self.y_train, X=self.X_train)
            ensemble_forecaster = search.best_forecaster_
            print("Optimal weights found via randomized search.")
            print(search.best_params_)

        # Save and store the best ensemble model
        self.best_forecasters['ensemble'] = ensemble_forecaster
        prediction = ensemble_forecaster.predict(X=self.X_valid, fh=self.fh)
        self.forecasters_results['ensemble'] = prediction
        self.results['Model'].append('ensemble')
        self.results['MAPE'].append(mean_absolute_percentage_error(self.y_valid, prediction))
        joblib.dump(ensemble_forecaster, os.path.join(self.save_dir, 'ensemble_model.pkl'))
        print("Ensemble model created and saved successfully.")
        
        return pd.DataFrame(self.results).set_index("Model").sort_values(by="MAPE")

    def make_predictions(self, model_indices=None, start_date="2024-01-01", length=30):
        """
        Makes predictions using a specified model on the test set.
        
        Parameters:
        - model_indices (str): The indices of the model to use for predictions.
        - start_date (str): The starting date for predictions (default is "2024-01-01").
        - length (int): The number of time steps to predict (default is 30).
        
        Returns:
        - pd.DataFrame: A DataFrame containing the predicted values.
        """
        # Check for available models
        if not self.best_forecasters:
            print("No fine-tuned models available. Load or fine-tune models before making predictions.")
            return
        
        print("Available models for prediction:")
        model_keys = list(self.best_forecasters.keys())
        for idx, model_name in enumerate(model_keys):
            print(f"{idx + 1}. {model_name}")
        
        if model_indices is None:
            print(f"Please specify model_indices parameter with the model numbers to include (comma-separated, no space) or 'all'")
            return
        elif model_indices.lower() == 'all':
            selected_models = model_keys
        else:
            try:
                indices = [int(x.strip()) - 1 for x in model_indices.split(',')]
                selected_models = [model_keys[i] for i in indices]
            except IndexError:
                print("Invalid model indices. No ensemble created.")
                return
        
        # Convert start_date to pandas datetime and create a forecast horizon
        start_date = pd.to_datetime(start_date)
        forecast_dates = pd.date_range(start=start_date, periods=length, freq="D")
        fh = ForecastingHorizon(forecast_dates, is_relative=False)
        forecast_dates_df = pd.DataFrame({'Datetime': forecast_dates})
        X_prediction = preprocess_kp_index(forecast_dates_df).set_index('Datetime')

        predictions_dict = {}
        fig = px.line(title=f"Predictions starting from {start_date}")

        # Loop through each model and generate predictions
        for model_name in selected_models:
            forecaster = self.best_forecasters.get(model_name)
            if not forecaster:
                raise ValueError(f"Model {model_name} is not available in best_forecasters.")
            
            # Make predictions
            predictions = forecaster.predict(X=X_prediction, fh=fh)
            predictions_dict[model_name] = predictions

            # Add predictions to plot
            fig.add_scatter(x=predictions.index, y=predictions, name=model_name)

        fig.show()

        return predictions_dict