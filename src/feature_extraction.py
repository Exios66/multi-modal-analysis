import numpy as np
import pandas as pd
import mne
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_eye_tracking_features(df):
    try:
        features = {}
        features['avg_fixation_duration'] = df['fixation_duration'].mean()
        features['max_fixation_duration'] = df['fixation_duration'].max()
        features['total_fixations'] = df.shape[0]
        features['avg_saccade_amplitude'] = df['saccade_amplitude'].mean()
        features['avg_pupil_size'] = df['pupil_size'].mean()
        logger.info("Eye tracking features extracted")
        return features
    except Exception as e:
        logger.error(f"Error extracting eye tracking features: {e}")
        raise

def extract_eeg_features(raw):
    try:
        features = {}
        # Compute Power Spectral Density (PSD) for standard frequency bands
        psd, freqs = mne.time_frequency.psd_welch(raw, n_fft=2048)
        # Define frequency bands
        bands = {'delta': (1, 4),
                 'theta': (4, 8),
                 'alpha': (8, 13),
                 'beta': (13, 30),
                 'gamma': (30, 50)}
        for band, (low, high) in bands.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            band_power = psd[:, idx_band].mean(axis=1)
            # Average across channels
            features[f'{band}_band_power'] = band_power.mean()
        logger.info("EEG features extracted")
        return features
    except Exception as e:
        logger.error(f"Error extracting EEG features: {e}")
        raise

def extract_vitals_features(df):
    try:
        features = {}
        features['avg_heart_rate'] = df['heart_rate'].mean()
        features['max_heart_rate'] = df['heart_rate'].max()
        features['min_heart_rate'] = df['heart_rate'].min()
        features['avg_blood_pressure'] = df['blood_pressure_systolic'].mean()
        logger.info("Vitals features extracted")
        return features
    except Exception as e:
        logger.error(f"Error extracting vitals features: {e}")
        raise

def extract_face_heatmap_features(df):
    try:
        features = {}
        # Example: Average intensity per facial region
        facial_regions = ['left_eye', 'right_eye', 'nose', 'mouth', 'forehead']
        for region in facial_regions:
            if region in df.columns:
                features[f'avg_{region}_intensity'] = df[region].mean()
        logger.info("Face heatmap features extracted")
        return features
    except Exception as e:
        logger.error(f"Error extracting face heatmap features: {e}")
        raise

def extract_survey_features(df):
    try:
        features = {}
        # Assuming survey responses are numerical scores
        for question_id, response_score in df.groupby('question_id')['response_score'].mean().items():
            features[f'question_{question_id}_avg_score'] = response_score
        logger.info("Survey features extracted")
        return features
    except Exception as e:
        logger.error(f"Error extracting survey features: {e}")
        raise
