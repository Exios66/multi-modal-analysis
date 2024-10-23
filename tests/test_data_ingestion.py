import pytest
import pandas as pd
import numpy as np
import os
from src.data_ingestion import (
    load_eye_tracking_data,
    load_eeg_data,
    load_survey_data,
    load_vitals_data,
    load_face_heatmap_data
)

@pytest.fixture
def sample_data_dir(tmp_path):
    """Create sample data files for testing."""
    # Create directories
    data_dir = tmp_path / "data"
    for subdir in ["eye_tracking", "eeg", "surveys", "vitals", "face_heatmap"]:
        os.makedirs(data_dir / subdir)
    
    # Create sample eye tracking data
    eye_data = pd.DataFrame({
        'timestamp': range(100),
        'fixation_duration': np.random.rand(100),
        'saccade_amplitude': np.random.rand(100),
        'pupil_size': np.random.rand(100)
    })
    eye_data.to_csv(data_dir / "eye_tracking" / "eye_data.csv", index=False)
    
    # Create sample vitals data
    vitals_data = pd.DataFrame({
        'timestamp': range(100),
        'heart_rate': np.random.randint(60, 100, 100),
        'blood_pressure_systolic': np.random.randint(100, 140, 100),
        'blood_pressure_diastolic': np.random.randint(60, 90, 100)
    })
    vitals_data.to_csv(data_dir / "vitals" / "vitals_data.csv", index=False)
    
    # Create sample survey data
    survey_data = pd.DataFrame({
        'question_id': range(10),
        'response_score': np.random.randint(1, 6, 10),
        'response_text': ['Sample response ' + str(i) for i in range(10)]
    })
    survey_data.to_csv(data_dir / "surveys" / "survey_data.csv", index=False)
    
    # Create sample face heatmap data
    face_data = pd.DataFrame({
        'timestamp': range(100),
        'left_eye': np.random.rand(100),
        'right_eye': np.random.rand(100),
        'nose': np.random.rand(100),
        'mouth': np.random.rand(100)
    })
    face_data.to_csv(data_dir / "face_heatmap" / "face_heatmap_data.csv", index=False)
    
    return data_dir

def test_load_eye_tracking_data(sample_data_dir):
    """Test loading eye tracking data."""
    data = load_eye_tracking_data(sample_data_dir / "eye_tracking" / "eye_data.csv")
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ['timestamp', 'fixation_duration', 'saccade_amplitude', 'pupil_size'])

def test_load_vitals_data(sample_data_dir):
    """Test loading vitals data."""
    data = load_vitals_data(sample_data_dir / "vitals" / "vitals_data.csv")
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ['timestamp', 'heart_rate', 'blood_pressure_systolic'])

def test_load_survey_data(sample_data_dir):
    """Test loading survey data."""
    data = load_survey_data(sample_data_dir / "surveys" / "survey_data.csv")
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ['question_id', 'response_score', 'response_text'])

def test_load_face_heatmap_data(sample_data_dir):
    """Test loading face heatmap data."""
    data = load_face_heatmap_data(sample_data_dir / "face_heatmap" / "face_heatmap_data.csv")
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ['timestamp', 'left_eye', 'right_eye', 'nose', 'mouth'])

def test_file_not_found():
    """Test handling of non-existent files."""
    with pytest.raises(Exception):
        load_eye_tracking_data("nonexistent_file.csv")

def test_invalid_data_format(tmp_path):
    """Test handling of invalid data format."""
    # Create invalid CSV file
    invalid_file = tmp_path / "invalid.csv"
    with open(invalid_file, 'w') as f:
        f.write("invalid,data\n")
    
    with pytest.raises(Exception):
        load_eye_tracking_data(invalid_file)
