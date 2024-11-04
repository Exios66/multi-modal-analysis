import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_eye_tracking_data():
    """Generate sample eye tracking data"""
    n_samples = 1000
    timestamps = [datetime.now() + timedelta(milliseconds=x) for x in range(n_samples)]
    
    data = {
        'timestamp': timestamps,
        'x_position': np.random.uniform(0, 1920, n_samples),
        'y_position': np.random.uniform(0, 1080, n_samples),
        'fixation_duration': np.random.gamma(2, 2, n_samples),
        'saccade_amplitude': np.random.lognormal(0, 0.5, n_samples),
        'pupil_size': np.random.normal(3, 0.5, n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv('eye_tracking_data.csv', index=False)

def generate_vitals_data():
    """Generate sample vitals data"""
    n_samples = 500
    timestamps = [datetime.now() + timedelta(seconds=x*2) for x in range(n_samples)]
    
    data = {
        'timestamp': timestamps,
        'heart_rate': np.random.normal(75, 10, n_samples),
        'blood_pressure_systolic': np.random.normal(120, 10, n_samples),
        'blood_pressure_diastolic': np.random.normal(80, 8, n_samples),
        'temperature': np.random.normal(37, 0.3, n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv('vitals_data.csv', index=False)

def generate_face_heatmap_data():
    """Generate sample face heatmap data"""
    n_samples = 300
    timestamps = [datetime.now() + timedelta(milliseconds=x*100) for x in range(n_samples)]
    
    data = {
        'timestamp': timestamps,
        'left_eye': np.random.uniform(0, 1, n_samples),
        'right_eye': np.random.uniform(0, 1, n_samples),
        'nose': np.random.uniform(0, 1, n_samples),
        'mouth': np.random.uniform(0, 1, n_samples),
        'forehead': np.random.uniform(0, 1, n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv('face_heatmap_data.csv', index=False)

if __name__ == "__main__":
    generate_eye_tracking_data()
    generate_vitals_data()
    generate_face_heatmap_data() 