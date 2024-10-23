import os
import pandas as pd
import mne
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_eye_tracking_data(path):
    try:
        df = pd.read_csv(path)
        logger.info(f"Eye tracking data loaded from {path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load eye tracking data: {e}")
        raise

def load_face_heatmap_data(path):
    try:
        df = pd.read_csv(path)
        logger.info(f"Face heatmap data loaded from {path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load face heatmap data: {e}")
        raise

def load_vitals_data(path):
    try:
        df = pd.read_csv(path)
        logger.info(f"Vitals data loaded from {path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load vitals data: {e}")
        raise

def load_eeg_data(path):
    try:
        raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
        logger.info(f"EEG data loaded from {path}")
        return raw
    except Exception as e:
        logger.error(f"Failed to load EEG data: {e}")
        raise

def load_survey_data(path):
    try:
        df = pd.read_csv(path)
        logger.info(f"Survey data loaded from {path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load survey data: {e}")
        raise

def load_json_data(path):
    try:
        with open(path, 'r') as file:
            data = json.load(file)
        logger.info(f"JSON data loaded from {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON data: {e}")
        raise

def get_data_files(data_dir):
    """Utility function to list all data files in a directory."""
    data_files = {}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv') or file.endswith('.edf') or file.endswith('.json'):
                key = os.path.splitext(file)[0]
                data_files[key] = os.path.join(root, file)
    return data_files
