import unittest
import pandas as pd
from src.data_ingestion import load_eye_tracking_data

class TestDataIngestion(unittest.TestCase):
    def test_load_eye_tracking_data(self):
        # Assuming a sample file 'sample_eye_data.csv' exists in 'tests/data/'
        sample_path = 'tests/data/sample_eye_data.csv'
        df = load_eye_tracking_data(sample_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn('fixation_duration', df.columns)

if __name__ == '__main__':
    unittest.main()
