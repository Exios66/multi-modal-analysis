import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from typing import Optional, Union, Dict, List
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define output directory
PLOTS_DIR = Path("output/plots")

def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 10),
    cmap: str = 'coolwarm'
) -> None:
    """
    Plot and optionally save a correlation heatmap.
    
    Parameters:
    - corr_matrix: pandas DataFrame representing the correlation matrix.
    - output_path: string or Path, path to save the plot. If None, display the plot.
    - figsize: tuple, size of the figure.
    - cmap: string, color map for the heatmap.
    
    Raises:
    - ValueError: If corr_matrix is not a DataFrame or is empty
    - IOError: If output_path is invalid or not writable
    """
    try:
        if not isinstance(corr_matrix, pd.DataFrame):
            raise ValueError("corr_matrix must be a pandas DataFrame")
        if corr_matrix.empty:
            raise ValueError("corr_matrix is empty")
            
        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            square=True,
            linewidths=.5,
            vmin=-1,
            vmax=1,
            center=0
        )
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        
        if output_path:
            output_path = PLOTS_DIR / Path(output_path)
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation heatmap saved to {output_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting correlation heatmap: {e}")
        plt.close()
        raise

def plot_feature_distributions(
    features_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    bins: int = 15,
    figsize: tuple = (15, 10)
) -> None:
    """
    Plot histograms for each feature to visualize distributions.
    
    Parameters:
    - features_df: pandas DataFrame containing features.
    - output_path: string or Path, path to save the plot. If None, display the plot.
    - bins: int, number of bins for histograms.
    - figsize: tuple, size of the figure.
    
    Raises:
    - ValueError: If features_df is not a DataFrame or is empty
    - IOError: If output_path is invalid or not writable
    """
    try:
        if not isinstance(features_df, pd.DataFrame):
            raise ValueError("features_df must be a pandas DataFrame")
        if features_df.empty:
            raise ValueError("features_df is empty")
            
        n_features = len(features_df.columns)
        n_cols = 4
        n_rows = int(np.ceil(n_features / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for idx, (col_name, data) in enumerate(features_df.items()):
            ax = axes[idx]
            data.hist(bins=bins, ax=ax)
            ax.set_title(col_name)
            ax.tick_params(labelrotation=45)
            
        # Hide empty subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
            
        plt.suptitle('Feature Distributions')
        plt.tight_layout()
        
        if output_path:
            output_path = PLOTS_DIR / Path(output_path)
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature distributions plot saved to {output_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting feature distributions: {e}")
        plt.close()
        raise

def plot_regression_results(
    regression_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot regression coefficients.
    
    Parameters:
    - regression_df: pandas DataFrame with 'Feature' and 'Coefficient' columns.
    - output_path: string or Path, path to save the plot. If None, display the plot.
    - figsize: tuple, size of the figure.
    
    Raises:
    - ValueError: If regression_df is not a DataFrame or missing required columns
    - IOError: If output_path is invalid or not writable
    """
    try:
        if not isinstance(regression_df, pd.DataFrame):
            raise ValueError("regression_df must be a pandas DataFrame")
        if not {'Feature', 'Coefficient'}.issubset(regression_df.columns):
            raise ValueError("regression_df must contain 'Feature' and 'Coefficient' columns")
            
        plt.figure(figsize=figsize)
        sns.barplot(
            x='Coefficient',
            y='Feature',
            data=regression_df.sort_values('Coefficient', ascending=False),
            palette='viridis'
        )
        plt.title('Regression Coefficients')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            output_path = PLOTS_DIR / Path(output_path)
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Regression coefficients plot saved to {output_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting regression results: {e}")
        plt.close()
        raise

def plot_time_series(
    data: pd.DataFrame,
    title: str = 'Time Series Data',
    xlabel: str = 'Time',
    ylabel: str = 'Value',
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 7)
) -> None:
    """
    Plot a time series.
    
    Parameters:
    - data: pandas DataFrame with a 'timestamp' column.
    - title: string, title of the plot.
    - xlabel: string, label for the x-axis.
    - ylabel: string, label for the y-axis.
    - output_path: string or Path, path to save the plot. If None, display the plot.
    - figsize: tuple, size of the figure.
    
    Raises:
    - ValueError: If data is not a DataFrame or missing timestamp column
    - IOError: If output_path is invalid or not writable
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        if 'timestamp' not in data.columns:
            raise ValueError("data must contain a 'timestamp' column")
            
        plt.figure(figsize=figsize)
        for column in data.columns:
            if column != 'timestamp':
                plt.plot(data['timestamp'], data[column], label=column, linewidth=2)
                
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            output_path = PLOTS_DIR / Path(output_path)
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Time series plot saved to {output_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting time series: {e}")
        plt.close()
        raise
