import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_noisy_data():
    """Generate sample data with noise and outliers"""
    n_samples = 1000
    timestamps = [datetime.now() + timedelta(milliseconds=x) for x in range(n_samples)]
    
    # Generate base signals
    clean_signal = np.sin(np.linspace(0, 10*np.pi, n_samples))
    
    # Add noise and outliers
    noise = np.random.normal(0, 0.2, n_samples)
    outliers = np.zeros(n_samples)
    outlier_idx = np.random.choice(n_samples, size=50, replace=False)
    outliers[outlier_idx] = np.random.uniform(-5, 5, 50)
    
    noisy_signal = clean_signal + noise + outliers
    
    # Create missing values
    missing_idx = np.random.choice(n_samples, size=100, replace=False)
    noisy_signal[missing_idx] = np.nan
    
    data = {
        'timestamp': timestamps,
        'clean_signal': clean_signal,
        'noisy_signal': noisy_signal,
        'categorical': np.random.choice(['A', 'B', 'C', None], n_samples)
    }
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Generate and save noisy data
    df = generate_noisy_data()
    df.to_csv('data/noisy_data.csv', index=False) 