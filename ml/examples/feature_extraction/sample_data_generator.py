import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mne
import os

def generate_eeg_sample():
    """Generate sample EEG data"""
    # Create sample EEG data
    sfreq = 256  # Sampling frequency
    t = np.linspace(0, 10, sfreq * 10)  # 10 seconds of data
    n_channels = 32
    
    # Generate random EEG-like signals
    data = np.random.randn(n_channels, len(t))
    ch_names = [f'EEG{i:03d}' for i in range(1, n_channels + 1)]
    ch_types = ['eeg'] * n_channels
    
    # Create MNE info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Create Raw object
    raw = mne.io.RawArray(data, info)
    
    return raw

def generate_survey_sample():
    """Generate sample survey data"""
    n_participants = 100
    n_questions = 10
    
    data = []
    for pid in range(n_participants):
        for qid in range(n_questions):
            response_time = np.random.uniform(1, 10)
            response_score = np.random.randint(1, 6)
            data.append({
                'participant_id': pid,
                'question_id': qid,
                'response_score': response_score,
                'response_time': response_time,
                'timestamp': datetime.now() + timedelta(seconds=qid)
            })
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate and save EEG data
    raw = generate_eeg_sample()
    raw.save('data/sample_eeg_raw.fif', overwrite=True)
    
    # Generate and save survey data
    survey_df = generate_survey_sample()
    survey_df.to_csv('data/sample_survey.csv', index=False) 