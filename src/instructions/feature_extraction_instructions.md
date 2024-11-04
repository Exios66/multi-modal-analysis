# Feature Extraction Module Instructions

## Overview

The `feature_extraction.py` module extracts relevant features from preprocessed data for each modality. It provides comprehensive feature extraction capabilities for eye tracking, face heatmap, vitals, EEG, and survey data.

## Features

- Eye tracking features:
  - Statistical measures (mean/max/std velocity)
  - Spatial dispersion metrics
  - Gaze pattern analysis
  - Fixation/saccade features

- Face heatmap features:
  - Intensity statistics
  - Spatial distribution metrics
  - Temporal intensity patterns
  - Region-based analysis

- Vitals features:
  - Basic statistics (mean, std, range)
  - Time-domain features
  - Trend analysis
  - Cross-correlation metrics

- EEG features:
  - Frequency band power (delta, theta, alpha, beta)
  - Channel-wise analysis
  - Power spectral density
  - Time-frequency features

- Survey features:
  - Response statistics
  - Pattern analysis
  - Entropy calculations
  - Distribution metrics

- Cross-modality features:
  - Temporal alignment
  - Feature correlation
  - Multimodal patterns
  - Joint distributions

## Function Documentation

### Main Functions

#### extract_multimodal_features(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]

Main function to extract features from all modalities.

Parameters:

- data_dict (Dict[str, pd.DataFrame]): Dictionary containing preprocessed data for each modality
  - Keys: modality names ('eye_tracking', 'face_heatmap', 'vitals', 'eeg', 'survey')
  - Values: pandas DataFrames with preprocessed data

Returns:

- Dict[str, pd.DataFrame]: Dictionary containing extracted features for each modality

#### extract_eye_tracking_features(df: pd.DataFrame) -> pd.DataFrame

Extracts features from eye tracking data.

Parameters:

- df (pd.DataFrame): Preprocessed eye tracking data
  - Required columns: 'velocity', 'x_position', 'y_position'

Returns:

- pd.DataFrame: Extracted eye tracking features

#### extract_face_heatmap_features(df: pd.DataFrame) -> pd.DataFrame

Extracts features from face heatmap data.

Parameters:

- df (pd.DataFrame): Preprocessed face heatmap data
  - Required columns: 'heatmap_values'

Returns:

- pd.DataFrame: Extracted face heatmap features

#### extract_vitals_features(df: pd.DataFrame) -> pd.DataFrame

Extracts features from vitals data.

Parameters:

- df (pd.DataFrame): Preprocessed vitals data
  - Required columns: 'heart_rate', 'blood_pressure', 'temperature'

Returns:

- pd.DataFrame: Extracted vitals features

#### extract_eeg_features(df: pd.DataFrame) -> pd.DataFrame

Extracts features from EEG data.

Parameters:

- df (pd.DataFrame): Preprocessed EEG data
  - Columns: EEG channels (excluding timestamp)

Returns:

- pd.DataFrame: Extracted EEG features

#### extract_survey_features(df: pd.DataFrame) -> pd.DataFrame

Extracts features from survey data.

Parameters:

- df (pd.DataFrame): Preprocessed survey data
  - Required columns: 'response'

Returns:

- pd.DataFrame: Extracted survey features

### Helper Functions

#### calculate_dispersion(x: pd.Series, y: pd.Series) -> float

Calculates spatial dispersion of gaze points.

#### calculate_trend(series: pd.Series) -> float

Calculates linear trend in time series data.

#### calculate_band_power(freqs: np.ndarray, psd: np.ndarray, low_freq: float, high_freq: float) -> float

Calculates power in specific frequency band for EEG analysis.

### Main Function
