import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_correlations(features_df, method='pearson'):
    """
    Compute correlation matrix for the features dataframe.
    
    Parameters:
    - features_df: pandas DataFrame containing features.
    - method: correlation method ('pearson', 'spearman', 'kendall')
    
    Returns:
    - Correlation matrix as pandas DataFrame.
    """
    try:
        corr_matrix = features_df.corr(method=method)
        logger.info(f"Correlation matrix computed using {method} method")
        return corr_matrix
    except Exception as e:
        logger.error(f"Error computing correlation matrix: {e}")
        raise

def identify_significant_correlations(corr_matrix, threshold=0.5, method='pearson'):
    """
    Identify correlations above a specified threshold.
    
    Parameters:
    - corr_matrix: pandas DataFrame representing the correlation matrix.
    - threshold: float, the correlation coefficient threshold.
    - method: string, correlation method used.
    
    Returns:
    - DataFrame of significant correlations.
    """
    try:
        significant = corr_matrix.where((corr_matrix.abs() >= threshold) & (corr_matrix.abs() != 1.0))
        significant = significant.stack().reset_index()
        significant.columns = ['Feature_1', 'Feature_2', 'Correlation']
        # Remove duplicate pairs
        significant = significant[significant['Feature_1'] < significant['Feature_2']]
        logger.info(f"Identified {significant.shape[0]} significant correlations using threshold {threshold}")
        return significant
    except Exception as e:
        logger.error(f"Error identifying significant correlations: {e}")
        raise

def perform_regression(features_df, target_feature):
    """
    Perform linear regression predicting target_feature from other features.
    
    Parameters:
    - features_df: pandas DataFrame containing features.
    - target_feature: string, the feature to predict.
    
    Returns:
    - Regression results as a pandas DataFrame.
    """
    from sklearn.linear_model import LinearRegression
    try:
        X = features_df.drop(columns=[target_feature]).values
        y = features_df[target_feature].values
        model = LinearRegression()
        model.fit(X, y)
        coefficients = pd.Series(model.coef_, index=features_df.drop(columns=[target_feature]).columns)
        intercept = model.intercept_
        logger.info(f"Performed linear regression for {target_feature}")
        return pd.DataFrame({'Feature': coefficients.index, 'Coefficient': coefficients.values})
    except Exception as e:
        logger.error(f"Error performing regression: {e}")
        raise
