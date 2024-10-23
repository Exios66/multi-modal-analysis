import os
import pandas as pd
import logging

# Import custom modules
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Define data directory
        data_dir = 'data'
        
        # Data Ingestion
        eye_data_path = os.path.join(data_dir, 'eye_tracking', 'eye_data.csv')
        eeg_data_path = os.path.join(data_dir, 'eeg', 'eeg_data.edf')
        survey_data_path = os.path.join(data_dir, 'surveys', 'survey_data.csv')
        vitals_data_path = os.path.join(data_dir, 'vitals', 'vitals_data.csv')
        face_heatmap_data_path = os.path.join(data_dir, 'face_heatmap', 'face_heatmap_data.csv')
        
        eye_data = load_eye_tracking_data(eye_data_path)
        eeg_data = load_eeg_data(eeg_data_path)
        survey_data = load_survey_data(survey_data_path)
        vitals_data = load_vitals_data(vitals_data_path)
        face_heatmap_data = load_face_heatmap_data(face_heatmap_data_path)
        
        # Data Preprocessing
        eye_data = preprocess_eye_tracking(eye_data)
        eeg_data = preprocess_eeg(eeg_data)
        survey_data = preprocess_survey(survey_data)
        vitals_data = preprocess_vitals(vitals_data)
        face_heatmap_data = preprocess_face_heatmap(face_heatmap_data)
        
        # Feature Extraction
        eye_features = extract_eye_tracking_features(eye_data)
        eeg_features = extract_eeg_features(eeg_data)
        survey_features = extract_survey_features(survey_data)
        vitals_features = extract_vitals_features(vitals_data)
        face_heatmap_features = extract_face_heatmap_features(face_heatmap_data)
        
        # Combine Features into a Single DataFrame
        combined_features = {**eye_features, **eeg_features, **survey_features, **vitals_features, **face_heatmap_features}
        features_df = pd.DataFrame([combined_features])
        logger.info(f"Combined features dataframe created with shape {features_df.shape}")
        
        # Correlation Analysis
        corr_matrix = compute_correlations(features_df, method='pearson')
        significant_corrs = identify_significant_correlations(corr_matrix, threshold=0.5, method='pearson')
        logger.info(f"Significant correlations:\n{significant_corrs}")
        
        # Regression Analysis (Example: Predicting avg_heart_rate)
        if 'avg_heart_rate' in features_df.columns:
            regression_results = perform_regression(features_df, target_feature='avg_heart_rate')
            logger.info(f"Regression results:\n{regression_results}")
        else:
            logger.warning("Target feature 'avg_heart_rate' not found in features. Skipping regression analysis.")
            regression_results = None
        
        # Visualization
        # Ensure output directories exist
        output_dir = 'output/plots'
        os.makedirs(output_dir, exist_ok=True)
        
        # Correlation Heatmap
        plot_correlation_heatmap(corr_matrix, output_path=os.path.join(output_dir, 'correlation_heatmap.png'))
        
        # Feature Distributions
        plot_feature_distributions(features_df, output_path=os.path.join(output_dir, 'feature_distributions.png'))
        
        # Regression Results
        if regression_results is not None:
            plot_regression_results(regression_results, output_path=os.path.join(output_dir, 'regression_coefficients.png'))
        
        # Optional: Time Series Plots (e.g., EEG Signals)
        # Convert EEG data to a DataFrame for plotting (simplified example)
        eeg_df = eeg_data.to_data_frame().reset_index()
        plot_time_series(eeg_df, title='EEG Time Series', xlabel='Time (s)', ylabel='Amplitude', output_path=os.path.join(output_dir, 'eeg_time_series.png'))
        
        logger.info("Application executed successfully.")
    
    except Exception as e:
        logger.error(f"An error occurred during application execution: {e}")

if __name__ == '__main__':
    main()
