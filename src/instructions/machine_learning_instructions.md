# Machine Learning Module Instructions

## Overview

The `machine_learning.py` module provides comprehensive functionality for training, evaluating, and persisting machine learning models with robust error handling and logging capabilities.

## Features

- Classification models:
  - Support for LogisticRegression and RandomForest classifiers
  - Hyperparameter tuning via GridSearchCV
  - Input validation and error handling
  - Missing value handling
  - Model evaluation metrics (classification report, confusion matrix)
  - Cross-validation scoring

- Regression models:
  - Support for LinearRegression and RandomForest regressors
  - Hyperparameter optimization
  - Input validation and error handling
  - Missing value handling
  - Model evaluation metrics (MSE, R-squared)
  - Cross-validation scoring

- Clustering capabilities:
  - K-Means clustering implementation
  - Feature scaling
  - Missing value handling
  - Cluster evaluation metrics
  - Configurable number of clusters

- Model evaluation:
  - Classification metrics:
    - Classification report
    - Confusion matrix
    - Cross-validation scores
  - Regression metrics:
    - Mean squared error
    - Root mean squared error
    - R-squared score
  - Clustering metrics:
    - Inertia (within-cluster sum of squares)

- Model persistence:
  - Save trained models to disk
  - Load models from disk
  - Error handling for file operations
  - Automatic file extension handling

## Main Functions

### Classification

#### train_classification_model(X, y, model_type='RandomForest', params=None)

Trains a classification model with input validation and error handling.

Parameters:

- X: pandas DataFrame, feature matrix
- y: pandas Series/array, target labels  
- model_type: str, 'LogisticRegression' or 'RandomForest'
- params: dict, hyperparameters for GridSearchCV

Returns:

- Pipeline: trained model with best parameters

### Regression

#### train_regression_model(X, y, model_type='RandomForest', params=None)

Trains a regression model with input validation and error handling.

Parameters:

- X: pandas DataFrame, feature matrix
- y: pandas Series/array, target variable
- model_type: str, 'LinearRegression' or 'RandomForest'
- params: dict, hyperparameters for GridSearchCV

Returns:

- Pipeline: trained model with best parameters

### Clustering

#### train_clustering_model(X, n_clusters=5, random_state=42)

Trains a K-Means clustering model with validation and error handling.

Parameters:

- X: pandas DataFrame, feature matrix
- n_clusters: int, number of clusters
- random_state: int, random seed

Returns:

- Tuple[KMeans, np.ndarray]: trained model and cluster labels

### Model Evaluation

#### evaluate_classification_model(model, X_test, y_test)

Evaluates classification model performance.

Parameters:

- model: trained classification Pipeline
- X_test: pandas DataFrame, test features
- y_test: pandas Series/array, true labels

Returns:

- Tuple[str, pd.DataFrame]: classification report and confusion matrix

#### evaluate_regression_model(model, X_test, y_test)

Evaluates regression model performance.

Parameters:

- model: trained regression Pipeline
- X_test: pandas DataFrame, test features
- y_test: pandas Series/array, true values

Returns:

- Tuple[float, float]: MSE and R-squared scores

### Model Persistence

#### save_model(model, filepath)

Saves trained model to disk.

Parameters:

- model: trained model object
- filepath: str, save location

#### load_model(filepath)

Loads trained model from disk.

Parameters:

- filepath: str, model file location

Returns:

- Loaded model object
