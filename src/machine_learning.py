import os
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
from typing import Dict, Tuple, Any, Optional, Union
import warnings

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_classification_model(
    X: pd.DataFrame, 
    y: Union[pd.Series, np.ndarray], 
    model_type: str = 'RandomForest',
    params: Optional[Dict[str, Any]] = None
) -> Pipeline:
    """
    Train a classification model with input validation and error handling.
    
    Parameters:
    - X: pandas DataFrame, feature matrix
    - y: pandas Series or array-like, target labels
    - model_type: string, type of model ('LogisticRegression', 'RandomForest')
    - params: dict, hyperparameters for GridSearchCV
    
    Returns:
    - best_model: trained Pipeline model with best parameters
    
    Raises:
    - ValueError: If input data is invalid or model type not supported
    - Exception: For other unexpected errors during training
    """
    try:
        # Input validation
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError("y must be a pandas Series or numpy array")
        if X.shape[0] != len(y):
            raise ValueError("X and y must have same number of samples")
        if X.empty:
            raise ValueError("Empty feature matrix provided")
        
        # Handle missing values
        if X.isnull().any().any():
            logger.warning("Missing values detected in feature matrix")
            X = X.fillna(X.mean())
        
        # Model selection
        if model_type == 'LogisticRegression':
            model = LogisticRegression(max_iter=1000, random_state=42)
            default_params = {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            }
        elif model_type == 'RandomForest':
            model = RandomForestClassifier(random_state=42)
            default_params = {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10]
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        # Training with or without hyperparameter tuning
        if params is None:
            params = default_params
            logger.info("Using default hyperparameter grid")
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid = GridSearchCV(
                pipeline, 
                params, 
                cv=5, 
                n_jobs=-1, 
                scoring='accuracy',
                verbose=1
            )
            grid.fit(X, y)
            best_model = grid.best_estimator_
            
        # Log training results
        logger.info(f"{model_type} trained successfully")
        logger.info(f"Best parameters: {grid.best_params_}")
        logger.info(f"Best cross-validation score: {grid.best_score_:.4f}")
        
        return best_model
        
    except Exception as e:
        logger.error(f"Error training classification model: {str(e)}")
        raise

def train_regression_model(
    X: pd.DataFrame, 
    y: Union[pd.Series, np.ndarray],
    model_type: str = 'RandomForest',
    params: Optional[Dict[str, Any]] = None
) -> Pipeline:
    """
    Train a regression model with input validation and error handling.
    
    Parameters:
    - X: pandas DataFrame, feature matrix
    - y: pandas Series or array-like, target variable
    - model_type: string, type of model ('LinearRegression', 'RandomForest')
    - params: dict, hyperparameters for GridSearchCV
    
    Returns:
    - best_model: trained Pipeline model with best parameters
    
    Raises:
    - ValueError: If input data is invalid or model type not supported
    - Exception: For other unexpected errors during training
    """
    try:
        # Input validation
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError("y must be a pandas Series or numpy array")
        if X.shape[0] != len(y):
            raise ValueError("X and y must have same number of samples")
        if X.empty:
            raise ValueError("Empty feature matrix provided")
            
        # Handle missing values
        if X.isnull().any().any():
            logger.warning("Missing values detected in feature matrix")
            X = X.fillna(X.mean())
            
        # Model selection
        if model_type == 'LinearRegression':
            model = LinearRegression()
            default_params = {
                'regressor__fit_intercept': [True, False],
                'regressor__normalize': [True, False]
            }
        elif model_type == 'RandomForest':
            model = RandomForestRegressor(random_state=42)
            default_params = {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__max_depth': [None, 10, 20],
                'regressor__min_samples_split': [2, 5, 10]
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        
        # Training with or without hyperparameter tuning
        if params is None:
            params = default_params
            logger.info("Using default hyperparameter grid")
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid = GridSearchCV(
                pipeline,
                params,
                cv=5,
                n_jobs=-1,
                scoring='neg_mean_squared_error',
                verbose=1
            )
            grid.fit(X, y)
            best_model = grid.best_estimator_
            
        # Log training results
        logger.info(f"{model_type} trained successfully")
        logger.info(f"Best parameters: {grid.best_params_}")
        logger.info(f"Best cross-validation MSE: {-grid.best_score_:.4f}")
        
        return best_model
        
    except Exception as e:
        logger.error(f"Error training regression model: {str(e)}")
        raise

def evaluate_classification_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, np.ndarray]
) -> Tuple[str, pd.DataFrame]:
    """
    Evaluate a classification model with detailed metrics.
    
    Parameters:
    - model: trained classification Pipeline model
    - X_test: pandas DataFrame, test feature matrix
    - y_test: pandas Series or array-like, true labels
    
    Returns:
    - report: string, detailed classification report
    - conf_matrix: pandas DataFrame, confusion matrix with labels
    
    Raises:
    - ValueError: If input data is invalid
    - Exception: For other unexpected errors during evaluation
    """
    try:
        # Input validation
        if not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_test must be a pandas DataFrame")
        if not isinstance(y_test, (pd.Series, np.ndarray)):
            raise ValueError("y_test must be a pandas Series or numpy array")
        if X_test.shape[0] != len(y_test):
            raise ValueError("X_test and y_test must have same number of samples")
            
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Generate evaluation metrics
        report = classification_report(y_test, y_pred)
        conf_matrix = pd.DataFrame(
            confusion_matrix(y_test, y_pred),
            index=[f'True_{label}' for label in np.unique(y_test)],
            columns=[f'Pred_{label}' for label in np.unique(y_test)]
        )
        
        # Log evaluation results
        logger.info("Classification model evaluation completed")
        logger.info(f"\nClassification Report:\n{report}")
        logger.info(f"\nConfusion Matrix:\n{conf_matrix}")
        
        return report, conf_matrix
        
    except Exception as e:
        logger.error(f"Error evaluating classification model: {str(e)}")
        raise

def evaluate_regression_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, np.ndarray]
) -> Tuple[float, float]:
    """
    Evaluate a regression model with detailed metrics.
    
    Parameters:
    - model: trained regression Pipeline model
    - X_test: pandas DataFrame, test feature matrix
    - y_test: pandas Series or array-like, true target values
    
    Returns:
    - mse: float, Mean Squared Error
    - r2: float, R-squared score
    
    Raises:
    - ValueError: If input data is invalid
    - Exception: For other unexpected errors during evaluation
    """
    try:
        # Input validation
        if not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_test must be a pandas DataFrame")
        if not isinstance(y_test, (pd.Series, np.ndarray)):
            raise ValueError("y_test must be a pandas Series or numpy array")
        if X_test.shape[0] != len(y_test):
            raise ValueError("X_test and y_test must have same number of samples")
            
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Log evaluation results
        logger.info("Regression model evaluation completed")
        logger.info(f"Mean Squared Error: {mse:.4f}")
        logger.info(f"Root Mean Squared Error: {rmse:.4f}")
        logger.info(f"R-squared Score: {r2:.4f}")
        
        return mse, r2
        
    except Exception as e:
        logger.error(f"Error evaluating regression model: {str(e)}")
        raise

def train_clustering_model(
    X: pd.DataFrame,
    n_clusters: int = 5,
    random_state: int = 42
) -> Tuple[KMeans, np.ndarray]:
    """
    Train a clustering model using K-Means with validation and error handling.
    
    Parameters:
    - X: pandas DataFrame, feature matrix
    - n_clusters: int, number of clusters
    - random_state: int, random seed
    
    Returns:
    - model: trained KMeans model
    - labels: array-like, cluster labels for each sample
    
    Raises:
    - ValueError: If input data is invalid
    - Exception: For other unexpected errors during training
    """
    try:
        # Input validation
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if X.empty:
            raise ValueError("Empty feature matrix provided")
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
            
        # Handle missing values
        if X.isnull().any().any():
            logger.warning("Missing values detected in feature matrix")
            X = X.fillna(X.mean())
            
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        model.fit(X_scaled)
        labels = model.labels_
        
        # Calculate metrics
        inertia = model.inertia_
        
        # Log training results
        logger.info(f"K-Means clustering completed with {n_clusters} clusters")
        logger.info(f"Inertia (within-cluster sum of squares): {inertia:.4f}")
        
        return model, labels
        
    except Exception as e:
        logger.error(f"Error training clustering model: {str(e)}")
        raise

def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to disk with error handling.
    
    Parameters:
    - model: trained model object
    - filepath: string, path to save the model
    
    Raises:
    - ValueError: If filepath is invalid
    - Exception: For other unexpected errors during saving
    """
    try:
        # Input validation
        if not filepath.endswith('.joblib'):
            filepath += '.joblib'
            
        # Save model
        joblib.dump(model, filepath)
        logger.info(f"Model successfully saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk with error handling.
    
    Parameters:
    - filepath: string, path to the saved model
    
    Returns:
    - model: loaded model object
    
    Raises:
    - FileNotFoundError: If model file doesn't exist
    - Exception: For other unexpected errors during loading
    """
    try:
        # Input validation
        if not filepath.endswith('.joblib'):
            filepath += '.joblib'
            
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        # Load model
        model = joblib.load(filepath)
        logger.info(f"Model successfully loaded from {filepath}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise