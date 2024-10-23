import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class DataConfig:
    eye_tracking_path: str = 'data/eye_tracking/eye_data.csv'
    eeg_path: str = 'data/eeg/eeg_data.edf'
    survey_path: str = 'data/surveys/survey_data.csv'
    vitals_path: str = 'data/vitals/vitals_data.csv'
    face_heatmap_path: str = 'data/face_heatmap/face_heatmap_data.csv'

@dataclass
class AnalysisConfig:
    correlation_threshold: float = 0.5
    correlation_method: str = 'pearson'
    n_clusters: int = 5
    random_state: int = 42
    max_features: int = 1000
    test_size: float = 0.2

@dataclass
class MLConfig:
    target_feature: str = 'avg_heart_rate'
    model_type: str = 'RandomForest'
    model_params: Dict = None
    save_model: bool = True
    model_path: str = 'models'

@dataclass
class OutputConfig:
    base_dir: str = 'output'
    plots_dir: str = 'plots'
    results_dir: str = 'results'
    models_dir: str = 'models'
    file_formats: Dict[str, str] = None

    def __post_init__(self):
        if self.file_formats is None:
            self.file_formats = {
                'plots': 'png',
                'results': 'csv',
                'models': 'joblib'
            }
        
        # Create output directories
        for dir_name in [self.base_dir, 
                        os.path.join(self.base_dir, self.plots_dir),
                        os.path.join(self.base_dir, self.results_dir),
                        os.path.join(self.base_dir, self.models_dir)]:
            os.makedirs(dir_name, exist_ok=True)

@dataclass
class Config:
    data: DataConfig = DataConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    ml: MLConfig = MLConfig()
    output: OutputConfig = OutputConfig()
    debug: bool = False

    def __post_init__(self):
        if self.ml.model_params is None:
            self.ml.model_params = {
                'RandomForest': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [10, 20, None]
                },
                'LogisticRegression': {
                    'classifier__C': [0.1, 1.0, 10.0],
                    'classifier__max_iter': [1000]
                }
            }

# Default configuration
DEFAULT_CONFIG = Config()
