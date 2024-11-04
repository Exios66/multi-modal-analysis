import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from scipy import stats
from scipy.signal import welch

logger = logging.getLogger(__name__)

def extract_multimodal_features(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Extract features from preprocessed multimodal data.
    
    Args:
        data_dict: Dictionary containing preprocessed data for each modality
        
    Returns:
        Dictionary containing extracted features for each modality
    """
    features = {}
    
    try:
        # Extract features from eye tracking data
        if 'eye_tracking' in data_dict:
            features['eye_tracking'] = extract_eye_tracking_features(data_dict['eye_tracking'])
            
        # Extract features from face heatmap data
        if 'face_heatmap' in data_dict:
            features['face_heatmap'] = extract_face_heatmap_features(data_dict['face_heatmap'])
            
        # Extract features from vitals data
        if 'vitals' in data_dict:
            features['vitals'] = extract_vitals_features(data_dict['vitals'])
            
        # Extract features from EEG data
        if 'eeg' in data_dict:
            features['eeg'] = extract_eeg_features(data_dict['eeg'])
            
        # Extract features from survey data
        if 'survey' in data_dict:
            features['survey'] = extract_survey_features(data_dict['survey'])
            
        logger.info("Successfully extracted features from all modalities")
        return features
        
    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        raise

def extract_eye_tracking_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from eye tracking data"""
    features = pd.DataFrame()
    
    # Statistical features
    features['mean_velocity'] = df['velocity'].mean()
    features['max_velocity'] = df['velocity'].max()
    features['std_velocity'] = df['velocity'].std()
    
    # Spatial features
    features['gaze_dispersion'] = calculate_dispersion(df['x_position'], df['y_position'])
    
    return features

def extract_face_heatmap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from face heatmap data"""
    features = pd.DataFrame()
    
    # Statistical features
    features['mean_intensity'] = df['heatmap_values'].mean()
    features['max_intensity'] = df['heatmap_values'].max()
    features['intensity_variance'] = df['heatmap_values'].var()
    
    return features

def extract_vitals_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from vitals data"""
    features = pd.DataFrame()
    
    for col in ['heart_rate', 'blood_pressure', 'temperature']:
        # Basic statistics
        features[f'{col}_mean'] = df[col].mean()
        features[f'{col}_std'] = df[col].std()
        features[f'{col}_range'] = df[col].max() - df[col].min()
        
        # Time-domain features
        features[f'{col}_trend'] = calculate_trend(df[col])
        
    return features

def extract_eeg_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from EEG data"""
    features = pd.DataFrame()
    
    for channel in df.columns[:-1]:  # Exclude timestamp column
        # Frequency domain features
        freqs, psd = welch(df[channel], fs=250)  # Assuming 250 Hz sampling rate
        
        # Calculate power in different frequency bands
        features[f'{channel}_delta'] = calculate_band_power(freqs, psd, 1, 4)
        features[f'{channel}_theta'] = calculate_band_power(freqs, psd, 4, 8)
        features[f'{channel}_alpha'] = calculate_band_power(freqs, psd, 8, 13)
        features[f'{channel}_beta'] = calculate_band_power(freqs, psd, 13, 30)
        
    return features

def extract_survey_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from survey data"""
    features = pd.DataFrame()
    
    # Calculate basic statistics of responses
    features['mean_response'] = df['response'].mean()
    features['response_std'] = df['response'].std()
    
    # Calculate response patterns
    features['response_entropy'] = stats.entropy(df['response'].value_counts(normalize=True))
    
    return features

# Helper functions
def calculate_dispersion(x: pd.Series, y: pd.Series) -> float:
    """Calculate spatial dispersion of gaze points"""
    return np.sqrt(x.var() + y.var())

def calculate_trend(series: pd.Series) -> float:
    """Calculate linear trend in time series"""
    x = np.arange(len(series))
    slope, _ = np.polyfit(x, series, 1)
    return slope

def calculate_band_power(freqs: np.ndarray, psd: np.ndarray, 
                        low_freq: float, high_freq: float) -> float:
    """Calculate power in specific frequency band"""
    idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
    return np.mean(psd[idx])
