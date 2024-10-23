import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json

# Import custom modules
from config import Config, DEFAULT_CONFIG
from src.data_ingestion import (
    load_eye_tracking_data,
    load_eeg_data,
    load_survey_data,
    load_vitals_data,
    load_face_heatmap_data
)
from src.data_preprocessing import (
    preprocess_eye_tracking,
    preprocess_eeg,
    preprocess_survey,
    preprocess_vitals,
    preprocess_face_heatmap,
    synchronize_data
)
from src.feature_extraction import (
    extract_eye_tracking_features,
    extract_eeg_features,
    extract_survey_features,
    extract_vitals_features,
    extract_face_heatmap_features
)
from src.correlation_analysis import (
    compute_correlations,
    identify_significant_correlations,
    perform_regression
)
from src.visualization import (
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_regression_results,
    plot_time_series
)
from src.machine_learning import (
    train_classification_model,
    train_regression_model,
    evaluate_classification_model,
    evaluate_regression_model,
    save_model
)

def setup_logging(config: Config) -> None:
    """Configure logging based on configuration."""
    log_level = logging.DEBUG if config.debug else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(config.output.base_dir, 'analysis.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

def validate_data(data: Dict[str, pd.DataFrame]) -> bool:
    """Validate loaded data for completeness and basic quality checks."""
    try:
        for name, df in data.items():
            if df is None or df.empty:
                logging.error(f"Data validation failed: {name} is empty")
                return False
            if df.isnull().values.any():
                logging.warning(f"Missing values found in {name}")
            if len(df) < 10:  # Minimum sample size check
                logging.warning(f"Small sample size in {name}: {len(df)} records")
        return True
    except Exception as e:
        logging.error(f"Data validation error: {e}")
        return False

def load_data(config: Config) -> Dict[str, pd.DataFrame]:
    """Load all data sources with timing and validation."""
    start_time = time.time()
    data = {}
    
    try:
        data['eye_tracking'] = load_eye_tracking_data(config.data.eye_tracking_path)
        data['eeg'] = load_eeg_data(config.data.eeg_path)
        data['survey'] = load_survey_data(config.data.survey_path)
        data['vitals'] = load_vitals_data(config.data.vitals_path)
        data['face_heatmap'] = load_face_heatmap_data(config.data.face_heatmap_path)
        
        logging.info(f"Data loading completed in {time.time() - start_time:.2f} seconds")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data: Dict[str, pd.DataFrame], config: Config) -> Dict[str, pd.DataFrame]:
    """Preprocess all data sources with timing."""
    start_time = time.time()
    
    try:
        processed_data = {
            'eye_tracking': preprocess_eye_tracking(data['eye_tracking']),
            'eeg': preprocess_eeg(data['eeg']),
            'survey': preprocess_survey(data['survey']),
            'vitals': preprocess_vitals(data['vitals']),
            'face_heatmap': preprocess_face_heatmap(data['face_heatmap'])
        }
        
        logging.info(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")
        return processed_data
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def extract_features(processed_data: Dict[str, pd.DataFrame], config: Config) -> pd.DataFrame:
    """Extract features from all processed data sources."""
    start_time = time.time()
    
    try:
        features = {
            **extract_eye_tracking_features(processed_data['eye_tracking']),
            **extract_eeg_features(processed_data['eeg']),
            **extract_survey_features(processed_data['survey']),
            **extract_vitals_features(processed_data['vitals']),
            **extract_face_heatmap_features(processed_data['face_heatmap'])
        }
        
        features_df = pd.DataFrame([features])
        logging.info(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
        return features_df
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        raise

def perform_analysis(features_df: pd.DataFrame, config: Config) -> Dict:
    """Perform correlation and regression analysis."""
    start_time = time.time()
    results = {}
    
    try:
        # Correlation analysis
        corr_matrix = compute_correlations(features_df, method=config.analysis.correlation_method)
        significant_corrs = identify_significant_correlations(
            corr_matrix, 
            threshold=config.analysis.correlation_threshold,
            method=config.analysis.correlation_method
        )
        
        # Regression analysis
        if config.ml.target_feature in features_df.columns:
            regression_results = perform_regression(features_df, target_feature=config.ml.target_feature)
        else:
            regression_results = None
            logging.warning(f"Target feature '{config.ml.target_feature}' not found")
        
        results = {
            'correlation_matrix': corr_matrix,
            'significant_correlations': significant_corrs,
            'regression_results': regression_results
        }
        
        logging.info(f"Analysis completed in {time.time() - start_time:.2f} seconds")
        return results
    except Exception as e:
        logging.error(f"Error performing analysis: {e}")
        raise

def train_models(features_df: pd.DataFrame, config: Config) -> Dict:
    """Train and evaluate machine learning models."""
    start_time = time.time()
    
    try:
        X = features_df.drop(columns=[config.ml.target_feature])
        y = features_df[config.ml.target_feature]
        
        # Train models
        model = train_regression_model(
            X, y,
            model_type=config.ml.model_type,
            params=config.ml.model_params.get(config.ml.model_type)
        )
        
        # Evaluate models
        mse, r2 = evaluate_regression_model(model, X, y)
        
        # Save model if configured
        if config.ml.save_model:
            model_path = os.path.join(
                config.output.base_dir,
                config.output.models_dir,
                f"{config.ml.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            )
            save_model(model, model_path)
        
        results = {
            'model': model,
            'metrics': {
                'mse': mse,
                'r2': r2
            }
        }
        
        logging.info(f"Model training completed in {time.time() - start_time:.2f} seconds")
        return results
    except Exception as e:
        logging.error(f"Error training models: {e}")
        raise

def generate_visualizations(features_df: pd.DataFrame, analysis_results: Dict, config: Config) -> None:
    """Generate and save all visualizations."""
    start_time = time.time()
    
    try:
        plots_dir = os.path.join(config.output.base_dir, config.output.plots_dir)
        
        # Correlation heatmap
        plot_correlation_heatmap(
            analysis_results['correlation_matrix'],
            output_path=os.path.join(plots_dir, f"correlation_heatmap.{config.output.file_formats['plots']}")
        )
        
        # Feature distributions
        plot_feature_distributions(
            features_df,
            output_path=os.path.join(plots_dir, f"feature_distributions.{config.output.file_formats['plots']}")
        )
        
        # Regression results
        if analysis_results['regression_results'] is not None:
            plot_regression_results(
                analysis_results['regression_results'],
                output_path=os.path.join(plots_dir, f"regression_results.{config.output.file_formats['plots']}")
            )
        
        logging.info(f"Visualization generation completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Error generating visualizations: {e}")
        raise

def save_results(features_df: pd.DataFrame, analysis_results: Dict, ml_results: Dict, config: Config) -> None:
    """Save all results to files."""
    try:
        results_dir = os.path.join(config.output.base_dir, config.output.results_dir)
        
        # Save features
        features_df.to_csv(
            os.path.join(results_dir, f"features.{config.output.file_formats['results']}"),
            index=False
        )
        
        # Save correlation results
        analysis_results['significant_correlations'].to_csv(
            os.path.join(results_dir, f"significant_correlations.{config.output.file_formats['results']}"),
            index=False
        )
        
        # Save ML metrics
        with open(os.path.join(results_dir, 'ml_metrics.json'), 'w') as f:
            json.dump(ml_results['metrics'], f, indent=4)
        
        logging.info("Results saved successfully")
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multi-modal Analysis Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def main():
    """Main execution pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = DEFAULT_CONFIG
    config.debug = args.debug
    
    # Setup logging
    setup_logging(config)
    
    try:
        # Record start time
        total_start_time = time.time()
        
        # Execute pipeline
        logging.info("Starting analysis pipeline")
        
        # Load data
        data = load_data(config)
        if not validate_data(data):
            raise ValueError("Data validation failed")
        
        # Preprocess data
        processed_data = preprocess_data(data, config)
        
        # Extract features
        features_df = extract_features(processed_data, config)
        
        # Perform analysis
        analysis_results = perform_analysis(features_df, config)
        
        # Train models
        ml_results = train_models(features_df, config)
        
        # Generate visualizations
        generate_visualizations(features_df, analysis_results, config)
        
        # Save results
        save_results(features_df, analysis_results, ml_results, config)
        
        # Log completion
        total_time = time.time() - total_start_time
        logging.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == '__main__':
    main()
