import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_correlated_data():
    """Generate sample data with known correlations"""
    n_samples = 1000
    
    # Generate base features
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    
    # Generate correlated features
    y1 = 0.8 * x1 + 0.2 * np.random.normal(0, 1, n_samples)  # Strong positive correlation
    y2 = -0.6 * x2 + 0.4 * np.random.normal(0, 1, n_samples)  # Moderate negative correlation
    y3 = np.random.normal(0, 1, n_samples)  # No correlation
    
    data = {
        'feature_1': x1,
        'feature_2': x2,
        'target_1': y1,
        'target_2': y2,
        'target_3': y3,
        'timestamp': [datetime.now() + timedelta(seconds=x) for x in range(n_samples)]
    }
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Generate and save correlated data
    df = generate_correlated_data()
    df.to_csv('data/correlated_data.csv', index=False) 