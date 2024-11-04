import pandas as pd
import numpy as np
import mne
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def preprocess_multimodal_data(data_dict: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Preprocess different types of multimodal data.
    
    Args:
        data_dict: Dictionary containing different types of raw data
        
    Returns:
        Dictionary containing preprocessed data for each modality
    """
    preprocessed_data = {}
    
    try:
        # Process eye tracking data
        if 'eye_tracking' in data_dict:
            preprocessed_data['eye_tracking'] = preprocess_eye_tracking(data_dict['eye_tracking'])
            
        # Process face heatmap data
        if 'face_heatmap' in data_dict:
            preprocessed_data['face_heatmap'] = preprocess_face_heatmap(data_dict['face_heatmap'])
            
        # Process vitals data
        if 'vitals' in data_dict:
            preprocessed_data['vitals'] = preprocess_vitals(data_dict['vitals'])
            
        # Process EEG data
        if 'eeg' in data_dict:
            preprocessed_data['eeg'] = preprocess_eeg(data_dict['eeg'])
            
        # Process survey data
        if 'survey' in data_dict:
            preprocessed_data['survey'] = preprocess_survey(data_dict['survey'])
            
        logger.info("Successfully preprocessed all data modalities")
        return preprocessed_data
        
    except Exception as e:
        logger.error(f"Error in preprocessing data: {str(e)}")
        raise

def preprocess_eye_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess eye tracking data"""
    # Remove outliers and interpolate missing values
    df = df.copy()
    for col in ['x_position', 'y_position']:
        df[col] = remove_outliers(df[col])
        df[col] = df[col].interpolate(method='linear')
    
    # Add velocity and acceleration features
    df['velocity'] = calculate_velocity(df['x_position'], df['y_position'], df['timestamp'])
    return df

def preprocess_face_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess face heatmap data"""
    df = df.copy()
    # Normalize heatmap values
    df['heatmap_values'] = normalize_series(df['heatmap_values'])
    return df

def preprocess_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess vitals data"""
    df = df.copy()
    # Remove outliers and smooth signals
    for col in ['heart_rate', 'blood_pressure', 'temperature']:
        df[col] = remove_outliers(df[col])
        df[col] = smooth_signal(df[col])
    return df

def preprocess_eeg(raw: mne.io.Raw) -> pd.DataFrame:
    """Preprocess EEG data"""
    # Apply basic EEG preprocessing
    raw_copy = raw.copy()
    raw_copy.filter(l_freq=1, h_freq=40)  # Basic frequency filtering
    raw_copy.notch_filter(freqs=50)  # Remove power line noise
    
    # Convert to DataFrame for easier handling
    data, times = raw_copy[:, :]
    df = pd.DataFrame(data.T, columns=raw_copy.ch_names)
    df['timestamp'] = times
    return df

def preprocess_survey(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess survey data"""
    df = df.copy()
    # Convert categorical responses to numerical
    df['response'] = pd.Categorical(df['response']).codes
    return df

# Helper functions
def remove_outliers(series: pd.Series, threshold: float = 3) -> pd.Series:
    """Remove outliers using z-score method"""
    z_scores = np.abs((series - series.mean()) / series.std())
    return series.mask(z_scores > threshold)

def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize values to 0-1 range"""
    return (series - series.min()) / (series.max() - series.min())

def smooth_signal(series: pd.Series, window: int = 5) -> pd.Series:
    """Smooth signal using rolling average"""
    return series.rolling(window=window, center=True).mean()

def calculate_velocity(x: pd.Series, y: pd.Series, t: pd.Series) -> pd.Series:
    """Calculate velocity from position data"""
    dx = x.diff()
    dy = y.diff()
    dt = pd.to_numeric(t).diff()
    return np.sqrt(dx**2 + dy**2) / dt