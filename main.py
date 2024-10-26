import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import logging
import json
import yaml
import multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from scipy import stats
import warnings

# Import custom modules
from config import Config, DEFAULT_CONFIG
from src.data_ingestion import *
from src.data_preprocessing import *
from src.feature_extraction import *
from src.correlation_analysis import *
from src.visualization import *
from src.machine_learning import *

# Enhanced configuration dataclass
@dataclass
class EnhancedConfig:
    """Enhanced configuration with additional parameters."""
    debug: bool = False
    n_jobs: int = mp.cpu_count()
    chunk_size: int = 10000
    cache_dir: str = ".cache"
    timeout: int = 3600
    retries: int = 3
    
    data: Dict[str, Any] = field(default_factory=lambda: {
        "paths": {},
        "formats": {},
        "sampling_rates": {},
        "validation_rules": {}
    })
    
    processing: Dict[str, Any] = field(default_factory=lambda: {
        "parallel": True,
        "batch_size": 1000,
        "memory_limit": "8G",
        "compression": "snappy"
    })
    
    features: Dict[str, Any] = field(default_factory=lambda: {
        "selection_method": "mutual_info",
        "n_features": 50,
        "importance_threshold": 0.01
    })
    
    ml: Dict[str, Any] = field(default_factory=lambda: {
        "cross_validation": {
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42
        },
        "hyperparameter_tuning": {
            "method": "bayesian",
            "n_trials": 100,
            "timeout": 3600
        }
    })

class DataValidator:
    """Validates data quality and integrity."""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.validation_rules = config.data.validation_rules
    
    def validate_schema(self, df: pd.DataFrame, dataset_name: str) -> bool:
        """Validate dataset schema against expected columns and types."""
        expected_schema = self.validation_rules.get(dataset_name, {}).get("schema", {})
        if not expected_schema:
            return True
            
        actual_dtypes = df.dtypes.to_dict()
        for col, dtype in expected_schema.items():
            if col not in actual_dtypes:
                logging.error(f"Missing column {col} in {dataset_name}")
                return False
            if not np.issubdtype(actual_dtypes[col], dtype):
                logging.error(f"Invalid dtype for {col} in {dataset_name}")
                return False
        return True
    
    def validate_ranges(self, df: pd.DataFrame, dataset_name: str) -> bool:
        """Validate numeric columns are within expected ranges."""
        range_rules = self.validation_rules.get(dataset_name, {}).get("ranges", {})
        if not range_rules:
            return True
            
        for col, (min_val, max_val) in range_rules.items():
            if col in df.columns:
                if not df[col].between(min_val, max_val).all():
                    logging.error(f"Values out of range for {col} in {dataset_name}")
                    return False
        return True

class DataProcessor:
    """Handles data processing with parallel execution capabilities."""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.executor = (ProcessPoolExecutor(max_workers=config.n_jobs) 
                        if config.processing.parallel else None)
    
    def process_in_parallel(self, func: callable, data: pd.DataFrame) -> pd.DataFrame:
        """Process data in parallel chunks."""
        if not self.config.processing.parallel:
            return func(data)
            
        chunks = np.array_split(data, max(1, len(data) // self.config.processing.batch_size))
        processed_chunks = list(self.executor.map(func, chunks))
        return pd.concat(processed_chunks, axis=0)
    
    def apply_feature_extraction(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract features with parallel processing support."""
        features_list = []
        for name, df in data.items():
            extract_func = globals()[f"extract_{name}_features"]
            features = self.process_in_parallel(extract_func, df)
            features_list.append(features)
        return pd.concat(features_list, axis=1)

class ModelTrainer:
    """Handles model training with advanced capabilities."""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.models = {}
        self.metrics = {}
    
    def create_pipeline(self, model_type: str) -> Pipeline:
        """Create a sklearn pipeline with preprocessing steps."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', get_model_class(model_type)(**self.config.ml.get(model_type, {})))
        ])
    
    def train_with_cv(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> Dict:
        """Train model with cross-validation and advanced metrics."""
        pipeline = self.create_pipeline(model_type)
        cv_params = self.config.ml.cross_validation
        
        scoring = {
            'mse': make_scorer(lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)),
            'r2': 'r2',
            'mae': 'neg_mean_absolute_error'
        }
        
        cv_results = cross_validate(
            pipeline, X, y,
            cv=cv_params.n_splits,
            scoring=scoring,
            n_jobs=self.config.n_jobs,
            return_estimator=True
        )
        
        self.models[model_type] = cv_results['estimator']
        self.metrics[model_type] = {
            metric: np.mean(scores) for metric, scores in cv_results.items()
            if metric.startswith('test_')
        }
        
        return {
            'model': pipeline,
            'cv_results': cv_results,
            'metrics': self.metrics[model_type]
        }

class ResultsManager:
    """Manages results storage and retrieval."""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.base_dir = Path(config.output.base_dir)
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories for results storage."""
        for dir_name in ['models', 'plots', 'results', 'logs']:
            (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def save_results(self, results: Dict[str, Any], timestamp: str):
        """Save results with proper formatting and compression."""
        for name, data in results.items():
            path = self.base_dir / 'results' / f"{name}_{timestamp}"
            
            if isinstance(data, pd.DataFrame):
                data.to_parquet(
                    f"{path}.parquet",
                    compression=self.config.processing.compression
                )
            elif isinstance(data, dict):
                with open(f"{path}.json", 'w') as f:
                    json.dump(data, f, indent=4)
            else:
                logging.warning(f"Unsupported data type for {name}")

def setup_logging(config: EnhancedConfig) -> None:
    """Enhanced logging setup with rotation and formatting."""
    log_dir = Path(config.output.base_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = log_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    handlers = [
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
    
    logging.basicConfig(
        level=logging.DEBUG if config.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Set specific logging levels for verbose libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

def parse_arguments() -> argparse.Namespace:
    """Enhanced argument parsing with additional options."""
    parser = argparse.ArgumentParser(description='Enhanced Multi-modal Analysis Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
    parser.add_argument('--n-jobs', type=int, help='Number of parallel jobs')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    return parser.parse_args()

def main():
    """Enhanced main execution pipeline."""
    # Parse arguments and load configuration
    args = parse_arguments()
    
    # Load and merge configurations
    config = EnhancedConfig()
    if args.config:
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
            config = EnhancedConfig(**config_dict)
    
    # Override with command line arguments
    if args.debug:
        config.debug = True
    if args.n_jobs:
        config.n_jobs = args.n_jobs
    if args.parallel:
        config.processing.parallel = True
    
    # Setup logging
    setup_logging(config)
    
    try:
        # Initialize components
        validator = DataValidator(config)
        processor = DataProcessor(config)
        trainer = ModelTrainer(config)
        results_manager = ResultsManager(config)
        
        # Record start time and create timestamp
        total_start_time = time.time()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        logging.info("Starting enhanced analysis pipeline")
        
        # Load and validate data
        data = load_data(config)
        for name, df in data.items():
            if not all([
                validator.validate_schema(df, name),
                validator.validate_ranges(df, name)
            ]):
                raise ValueError(f"Data validation failed for {name}")
        
        # Process data and extract features
        processed_data = processor.process_in_parallel(preprocess_data, data)
        features_df = processor.apply_feature_extraction(processed_data)
        
        # Perform analysis
        analysis_results = perform_analysis(features_df, config)
        
        # Train and evaluate models
        ml_results = {}
        for model_type in config.ml.model_types:
            ml_results[model_type] = trainer.train_with_cv(
                features_df.drop(columns=[config.ml.target_feature]),
                features_df[config.ml.target_feature],
                model_type
            )
        
        # Generate visualizations
        generate_visualizations(features_df, analysis_results, ml_results, config)
        
        # Save results
        results = {
            'features': features_df,
            'analysis': analysis_results,
            'ml_results': ml_results
        }
        results_manager.save_results(results, timestamp)
        
        # Log completion
        total_time = time.time() - total_start_time
        logging.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        raise
    finally:
        if hasattr(processor, 'executor') and processor.executor:
            processor.executor.shutdown()

if __name__ == '__main__':
    main()