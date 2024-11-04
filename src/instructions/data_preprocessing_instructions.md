# Data Preprocessing Module Instructions

## Overview

The `data_preprocessing.py` module handles the preprocessing of various data types in the multimodal analysis pipeline. This module provides comprehensive data cleaning and preparation functionality to ensure data quality and consistency before analysis.

## Features

- Preprocessing for multiple data modalities:
  - Eye tracking data: Fixation filtering, saccade detection, blink removal
  - Face heatmap data: Normalization, artifact removal, temporal smoothing
  - Vitals data: Signal filtering, peak detection, baseline correction
  - EEG data: Bandpass filtering, ICA cleaning, epoch rejection
  - Survey data: Response validation, encoding, missing value imputation
  - JSON data: Schema validation, flattening, type conversion

- Outlier removal:
  - Statistical methods (Z-score, IQR)
  - Domain-specific thresholds
  - Isolation Forest for multivariate detection
  - Local Outlier Factor (LOF)

- Signal smoothing:
  - Moving average filters
  - Savitzky-Golay filtering
  - Gaussian smoothing
  - Exponential smoothing
  - Kalman filtering

- Data normalization:
  - Min-max scaling
  - Z-score standardization  
  - Robust scaling
  - Quantile transformation
  - Power transformation

- Missing value handling:
  - Multiple imputation
  - Forward/backward fill
  - Interpolation methods
  - Mean/median/mode imputation
  - KNN imputation

## Function Documentation

### Main Functions

#### preprocess_data(data: Dict, modality: str, **kwargs) -> Dict

Main function to preprocess data based on modality type.

Parameters:

- data (Dict): Input data dictionary containing raw data
- modality (str): Type of data ('eye_tracking', 'face_heatmap', 'vitals', 'eeg', 'survey', 'json')
- **kwargs: Additional preprocessing parameters

Returns:

- Dict: Preprocessed data dictionary

#### remove_outliers(data: np.ndarray, method: str = 'zscore', threshold: float = 3.0) -> np.ndarray

Detects and removes outliers using specified method.

Parameters:

- data (np.ndarray): Input data array
- method (str): Outlier detection method ('zscore', 'iqr', 'isolation_forest', 'lof')
- threshold (float): Threshold for outlier detection

Returns:

- np.ndarray: Data with outliers removed

#### smooth_signal(data: np.ndarray, method: str = 'moving_avg', window: int = 5) -> np.ndarray

Applies signal smoothing using specified method.

Parameters:

- data (np.ndarray): Input signal data
- method (str): Smoothing method ('moving_avg', 'savgol', 'gaussian', 'exp', 'kalman')
- window (int): Window size for smoothing

Returns:

- np.ndarray: Smoothed signal data

#### normalize_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray

Normalizes data using specified method.

Parameters:

- data (np.ndarray): Input data array
- method (str): Normalization method ('minmax', 'zscore', 'robust', 'quantile', 'power')

Returns:

- np.ndarray: Normalized data array

#### impute_missing(data: np.ndarray, method: str = 'mean') -> np.ndarray

Handles missing values using specified imputation method.

Parameters:

- data (np.ndarray): Input data with missing values
- method (str): Imputation method ('mean', 'median', 'mode', 'knn', 'interpolate')

Returns:

- np.ndarray: Data with imputed missing values
