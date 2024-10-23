import pandas as pd
import mne
import logging
from functools import reduce

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_eye_tracking(df):
    try:
        initial_shape = df.shape
        df.fillna(method='ffill', inplace=True)
        df.dropna(inplace=True)
        logger.info(f"Eye tracking data preprocessed: {initial_shape} -> {df.shape}")
        # Additional preprocessing steps (e.g., filtering out unrealistic fixations)
        return df
    except Exception as e:
        logger.error(f"Error preprocessing eye tracking data: {e}")
        raise

def preprocess_eeg(raw):
    try:
        raw = raw.copy().filter(l_freq=1., h_freq=50.)
        raw = raw.copy().resample(256)  # Ensure consistent sampling rate
        raw = raw.apply_hilbert(envelope=True)  # Example preprocessing step
        logger.info("EEG data preprocessed")
        # Additional preprocessing steps (e.g., artifact removal)
        return raw
    except Exception as e:
        logger.error(f"Error preprocessing EEG data: {e}")
        raise

def preprocess_vitals(df):
    try:
        initial_shape = df.shape
        df.fillna(method='ffill', inplace=True)
        df.dropna(inplace=True)
        logger.info(f"Vitals data preprocessed: {initial_shape} -> {df.shape}")
        # Additional preprocessing steps (e.g., outlier removal)
        return df
    except Exception as e:
        logger.error(f"Error preprocessing vitals data: {e}")
        raise

def preprocess_face_heatmap(df):
    try:
        initial_shape = df.shape
        df.fillna(0, inplace=True)  # Assuming missing values can be set to zero
        logger.info(f"Face heatmap data preprocessed: {initial_shape} -> {df.shape}")
        # Additional preprocessing steps
        return df
    except Exception as e:
        logger.error(f"Error preprocessing face heatmap data: {e}")
        raise

def preprocess_survey(df):
    try:
        initial_shape = df.shape
        df.fillna(df.mean(), inplace=True)  # Assuming numeric responses
        logger.info(f"Survey data preprocessed: {initial_shape} -> {df.shape}")
        # Additional preprocessing steps (e.g., encoding categorical responses)
        return df
    except Exception as e:
        logger.error(f"Error preprocessing survey data: {e}")
        raise

def synchronize_data(dfs, on='timestamp'):
    """
    Synchronize multiple dataframes based on a common timestamp.
    
    Parameters:
    - dfs: list of pandas DataFrames
    - on: column name to merge on
    
    Returns:
    - Merged DataFrame
    """
    try:
        df_merged = reduce(lambda left, right: pd.merge(left, right, on=on, how='inner'), dfs)
        logger.info(f"Data synchronized on '{on}' with shape {df_merged.shape}")
        return df_merged
    except Exception as e:
        logger.error(f"Error synchronizing data: {e}")
        raise
