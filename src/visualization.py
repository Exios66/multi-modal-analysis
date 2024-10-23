import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_correlation_heatmap(corr_matrix, output_path=None, figsize=(12, 10), cmap='coolwarm'):
    """
    Plot and optionally save a correlation heatmap.
    
    Parameters:
    - corr_matrix: pandas DataFrame representing the correlation matrix.
    - output_path: string, path to save the plot. If None, display the plot.
    - figsize: tuple, size of the figure.
    - cmap: string, color map for the heatmap.
    """
    try:
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, square=True, linewidths=.5)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Correlation heatmap saved to {output_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting correlation heatmap: {e}")
        raise

def plot_feature_distributions(features_df, output_path=None, bins=15, figsize=(15, 10)):
    """
    Plot histograms for each feature to visualize distributions.
    
    Parameters:
    - features_df: pandas DataFrame containing features.
    - output_path: string, path to save the plot. If None, display the plot.
    - bins: int, number of bins for histograms.
    - figsize: tuple, size of the figure.
    """
    try:
        features_df.hist(bins=bins, figsize=figsize, layout=(int(np.ceil(len(features_df.columns)/4)), 4)))
        plt.suptitle('Feature Distributions')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Feature distributions plot saved to {output_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting feature distributions: {e}")
        raise

def plot_regression_results(regression_df, output_path=None, figsize=(10, 6)):
    """
    Plot regression coefficients.
    
    Parameters:
    - regression_df: pandas DataFrame with 'Feature' and 'Coefficient' columns.
    - output_path: string, path to save the plot. If None, display the plot.
    - figsize: tuple, size of the figure.
    """
    try:
        plt.figure(figsize=figsize)
        sns.barplot(x='Coefficient', y='Feature', data=regression_df.sort_values('Coefficient', ascending=False))
        plt.title('Regression Coefficients')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Regression coefficients plot saved to {output_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting regression results: {e}")
        raise

def plot_time_series(data, title='Time Series Data', xlabel='Time', ylabel='Value', output_path=None, figsize=(14, 7)):
    """
    Plot a time series.
    
    Parameters:
    - data: pandas DataFrame with a 'timestamp' column.
    - title: string, title of the plot.
    - xlabel: string, label for the x-axis.
    - ylabel: string, label for the y-axis.
    - output_path: string, path to save the plot. If None, display the plot.
    - figsize: tuple, size of the figure.
    """
    try:
        plt.figure(figsize=figsize)
        for column in data.columns:
            if column != 'timestamp':
                plt.plot(data['timestamp'], data[column], label=column)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Time series plot saved to {output_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting time series: {e}")
        raise
