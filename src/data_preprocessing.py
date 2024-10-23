import pandas as pd
import mne
import logging
from functools import reduce
import nltk
from nltk.corpus import stopwords
import string
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize SpaCy
nlp = spacy.load('en_core_web_sm')

# Download NLTK stopwords if not already
nltk.download('stopwords')

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
        # Handling numerical responses
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
        
        # Handling textual responses if any
        if 'response_text' in df.columns:
            df['response_text'] = df['response_text'].apply(clean_text)
            df['response_text'] = df['response_text'].fillna('')
        
        logger.info(f"Survey data preprocessed: {initial_shape} -> {df.shape}")
        # Additional preprocessing steps (e.g., encoding categorical responses)
        return df
    except Exception as e:
        logger.error(f"Error preprocessing survey data: {e}")
        raise

def clean_text(text):
    try:
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatization using SpaCy
        doc = nlp(' '.join(tokens))
        lemmatized = ' '.join([token.lemma_ for token in doc])
        return lemmatized
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return ''

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