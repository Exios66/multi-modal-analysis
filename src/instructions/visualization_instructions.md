# Visualization Module Instructions

## Overview

The `visualization.py` module provides comprehensive data visualization capabilities for multimodal data analysis results. It enables clear and informative visualization of statistical relationships, distributions, time series data, and analysis outcomes.

## Features

- Correlation heatmaps:
  - Visualize correlation matrices between features
  - Customizable color schemes and annotations
  - Adjustable figure size and layout
  - Optional saving to file
  - Automatic value annotations with 2 decimal precision
  - Square aspect ratio for readability
  - Center color scale at 0 with -1 to 1 range
  - Configurable linewidth between cells

- Feature distribution plots:
  - Histogram plots for each feature
  - Configurable number of bins
  - Automatic subplot layout with 4 columns
  - Feature name labels and 45-degree rotated ticks
  - Automatic hiding of empty subplots
  - Tight layout with optimized spacing
  - Optional saving to file

- Time series visualization:
  - Multi-line plots for temporal data
  - Customizable line styles and colors
  - Automatic legend generation with outside placement
  - Semi-transparent grid overlay
  - Configurable axis labels and title
  - Line width of 2 for visibility
  - Optional saving to file

- Regression result plots:
  - Bar plots of regression coefficients
  - Sorted coefficient visualization (descending)
  - Vertical reference line at zero
  - Viridis color palette
  - Feature names on y-axis
  - Optional saving to file

- Plot customization options:
  - Figure size adjustment via tuple parameters
  - Color scheme selection through cmap parameter
  - Title and axis label customization
  - Legend positioning outside plot area
  - 300 DPI resolution for saved files
  - Tight layout with optimized spacing

- File handling features:
  - Automatic creation of output/plots directory
  - Configurable file paths with Path objects
  - Parent directory creation if needed
  - High-resolution PNG exports
  - Informative logging of save operations
  - Error handling for invalid paths

## Function Documentation

### plot_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: Optional[Union[str, Path]] = None, figsize: tuple = (12, 10), cmap: str = 'coolwarm') -> None

Creates a correlation heatmap visualization.

Parameters:

- corr_matrix (pd.DataFrame): Correlation matrix DataFrame
- output_path (Optional[Union[str, Path]]): Path to save plot, if None displays plot
- figsize (tuple): Figure dimensions as (width, height)
- cmap (str): Matplotlib colormap name

Returns: None

Raises:

- ValueError: If corr_matrix is not DataFrame or empty
- IOError: If output path is invalid

### plot_feature_distributions(features_df: pd.DataFrame, output_path: Optional[Union[str, Path]] = None, bins: int = 15, figsize: tuple = (15, 10)) -> None

Generates distribution plots for each feature.

Parameters:

- features_df (pd.DataFrame): Feature data DataFrame
- output_path (Optional[Union[str, Path]]): Path to save plot, if None displays plot
- bins (int): Number of histogram bins
- figsize (tuple): Figure dimensions as (width, height)

Returns: None

Raises:

- ValueError: If features_df is not DataFrame or empty
- IOError: If output path is invalid

### plot_regression_results(regression_df: pd.DataFrame, output_path: Optional[Union[str, Path]] = None, figsize: tuple = (10, 6)) -> None

Visualizes regression coefficient values.

Parameters:

- regression_df (pd.DataFrame): DataFrame with Feature and Coefficient columns
- output_path (Optional[Union[str, Path]]): Path to save plot, if None displays plot
- figsize (tuple): Figure dimensions as (width, height)

Returns: None

Raises:

- ValueError: If regression_df missing required columns
- IOError: If output path is invalid

### plot_time_series(data: pd.DataFrame, title: str = 'Time Series Data', xlabel: str = 'Time', ylabel: str = 'Value', output_path: Optional[Union[str, Path]] = None, figsize: tuple = (14, 7)) -> None

Creates time series visualizations.

Parameters:

- data (pd.DataFrame): DataFrame with timestamp column and data series
- title (str): Plot title
- xlabel (str): X-axis label
- ylabel (str): Y-axis label
- output_path (Optional[Union[str, Path]]): Path to save plot, if None displays plot
- figsize (tuple): Figure dimensions as (width, height)

Returns: None

Raises:

- ValueError: If data missing timestamp column
- IOError: If output path is invalid

## Usage Examples
