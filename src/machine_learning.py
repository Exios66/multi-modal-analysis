import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_classification_model(X, y, model_type='RandomForest', params=None):
    """
    Train a classification model.
    
    Parameters:
    - X: pandas DataFrame, feature matrix
    - y: pandas Series or array-like, target labels
    - model_type: string, type of model ('LogisticRegression', 'RandomForest')
    - params: dict, hyperparameters for GridSearchCV
    
    Returns:
    - best_model: trained model with best parameters
    """
    try:
        if model_type == 'LogisticRegression':
            model = LogisticRegression(max_iter=1000)
        elif model_type == 'RandomForest':
            model = RandomForestClassifier()
        else:
            raise ValueError("Unsupported model type")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        if params:
            grid = GridSearchCV(pipeline, params, cv=5, n_jobs=-1, scoring='accuracy')
            grid.fit(X, y)
            best_model = grid.best_estimator_
            logger.info(f"{model_type} trained with best parameters: {grid.best_params_}")
        else:
            pipeline.fit(X, y)
            best_model = pipeline
            logger.info(f"{model_type} trained without hyperparameter tuning")
        
        return best_model
    except Exception as e:
        logger.error(f"Error training classification model: {e}")
        raise

def train_regression_model(X, y, model_type='RandomForest', params=None):
    """
    Train a regression model.
    
    Parameters:
    - X: pandas DataFrame, feature matrix
    - y: pandas Series or array-like, target variable
    - model_type: string, type of model ('LinearRegression', 'RandomForest')
    - params: dict, hyperparameters for GridSearchCV
    
    Returns:
    - best_model: trained model with best parameters
    """
    try:
        if model_type == 'LinearRegression':
            model = LinearRegression()
        elif model_type == 'RandomForest':
            model = RandomForestRegressor()
        else:
            raise ValueError("Unsupported model type")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        
        if params:
            grid = GridSearchCV(pipeline, params, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
            grid.fit(X, y)
            best_model = grid.best_estimator_
            logger.info(f"{model_type} trained with best parameters: {grid.best_params_}")
        else:
            pipeline.fit(X, y)
            best_model = pipeline
            logger.info(f"{model_type} trained without hyperparameter tuning")
        
        return best_model
    except Exception as e:
        logger.error(f"Error training regression model: {e}")
        raise

def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate a classification model.
    
    Parameters:
    - model: trained classification model
    - X_test: pandas DataFrame, test feature matrix
    - y_test: pandas Series or array-like, true labels
    
    Returns:
    - report: string, classification report
    - conf_matrix: pandas DataFrame, confusion matrix
    """
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred))
        logger.info("Classification model evaluated")
        return report, conf_matrix
    except Exception as e:
        logger.error(f"Error evaluating classification model: {e}")
        raise

def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluate a regression model.
    
    Parameters:
    - model: trained regression model
    - X_test: pandas DataFrame, test feature matrix
    - y_test: pandas Series or array-like, true target values
    
    Returns:
    - mse: float, Mean Squared Error
    - r2: float, R-squared score
    """
    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info("Regression model evaluated")
        return mse, r2
    except Exception as e:
        logger.error(f"Error evaluating regression model: {e}")
        raise

def train_clustering_model(X, n_clusters=5, random_state=42):
    """
    Train a clustering model using K-Means.
    
    Parameters:
    - X: pandas DataFrame, feature matrix
    - n_clusters: int, number of clusters
    - random_state: int, random seed
    
    Returns:
    - model: trained KMeans model
    - labels: array-like, cluster labels for each sample
    """
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
        model.fit(X_scaled)
        labels = model.labels_
        logger.info(f"K-Means clustering performed with {n_clusters} clusters")
        return model, labels
    except Exception as e:
        logger.error(f"Error training clustering model: {e}")
        raise

def save_model(model, filepath):
    """
    Save a trained model to disk.
    
    Parameters:
    - model: trained model
    - filepath: string, path to save the model
    """
    try:
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def load_model(filepath):
    """
    Load a trained model from disk.
    
    Parameters:
    - filepath: string, path to the saved model
    
    Returns:
    - model: loaded model
    """
    try:
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise