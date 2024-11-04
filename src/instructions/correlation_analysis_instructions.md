# Correlation Analysis Module Instructions

## Overview

The `correlation_analysis.py` module analyzes relationships between features from different data modalities. It provides comprehensive correlation analysis capabilities including correlation matrices, cross-modality correlations, feature importance scoring, and mutual information calculations.

## Features

- Correlation matrix calculation:
  - Spearman correlation between all features
  - Handles missing values
  - Returns symmetric correlation matrix
  - Values range from -1 to 1

- Cross-modality correlation analysis:
  - Pairwise correlations between modalities
  - Average correlation strength metrics
  - Modality relationship assessment
  - Automatic handling of different feature counts

- Feature importance scoring:
  - Based on correlation strength
  - Mean absolute correlation per feature
  - Sorted importance rankings
  - Identifies key predictive features

- Mutual information calculation:
  - Information theoretic relationship measure
  - Handles continuous and discrete variables
  - Binning for continuous variables
  - Normalized mutual information scores

- Error handling and logging:
  - Comprehensive error messages
  - Operation logging
  - Input validation
  - Exception handling

## Function Documentation

### analyze_multimodal_correlations(features_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]

Main function to analyze correlations between features from different modalities.

Parameters:

- features_dict (Dict[str, pd.DataFrame]): Dictionary mapping modality names to feature DataFrames

Returns:

- Dict[str, Any]: Dictionary containing:
  - correlation_matrix: Full correlation matrix
  - cross_modality_correlations: Correlations between modalities
  - feature_importance: Feature importance scores
  - modality_relationships: Detailed modality relationship metrics

Raises:

- ValueError: If features_dict is empty or invalid
- TypeError: If input types are incorrect

### combine_features(features_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame

Combines features from all modalities into a single DataFrame.

Parameters:

- features_dict (Dict[str, pd.DataFrame]): Dictionary of modality features

Returns:

- pd.DataFrame: Combined features with modality prefixes

### calculate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame

Calculates the correlation matrix for all features.

Parameters:

- df (pd.DataFrame): Input feature DataFrame

Returns:

- pd.DataFrame: Correlation matrix

### analyze_cross_modality_correlations(features_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]

Analyzes correlations between different modalities.

Parameters:

- features_dict (Dict[str, pd.DataFrame]): Dictionary of modality features

Returns:

- Dict[str, float]: Cross-modality correlation scores

### calculate_feature_importance(df: pd.DataFrame) -> pd.Series

Calculates importance of each feature based on correlation strength.

Parameters:

- df (pd.DataFrame): Input feature DataFrame

Returns:

- pd.Series: Feature importance scores

### analyze_modality_relationships(features_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]

Analyzes detailed relationships between modalities.

Parameters:

- features_dict (Dict[str, pd.DataFrame]): Dictionary of modality features

Returns:

- Dict[str, Any]: Modality relationship metrics

### calculate_modality_correlation(df1: pd.DataFrame, df2: pd.DataFrame) -> float

Calculates average correlation between two modalities.

Parameters:

- df1 (pd.DataFrame): First modality features
- df2 (pd.DataFrame): Second modality features

Returns:

- float: Average correlation score

### calculate_mutual_information(df1: pd.DataFrame, df2: pd.DataFrame) -> float

Calculates mutual information between two modalities.

Parameters:

- df1 (pd.DataFrame): First modality features
- df2 (pd.DataFrame): Second modality features

Returns:

- float: Average mutual information score

### calculate_mi_score(x: pd.Series, y: pd.Series, bins: int = 10) -> float

Calculates mutual information score between two variables.

Parameters:

- x (pd.Series): First variable
- y (pd.Series): Second variable
- bins (int): Number of bins for continuous variables

Returns:

- float: Mutual information score

### Main Function

The main function `analyze_multimodal_correlations()` orchestrates the complete correlation analysis workflow:

1. Input validation:
   - Validates features_dict contains valid DataFrames
   - Checks for consistent indices across modalities
   - Verifies non-empty DataFrames

2. Correlation analysis:
   - Calculates correlation matrices for each modality
   - Computes cross-modality correlations
   - Generates feature importance scores
   - Calculates mutual information between modalities

3. Results compilation:
   - Correlation matrices stored in nested dictionary
   - Cross-modality correlation scores
   - Feature importance rankings
   - Mutual information scores
   - Summary statistics and metrics

4. Error handling:
   - Comprehensive input validation
   - Exception handling with informative messages
   - Logging of analysis steps and results

Returns a dictionary containing all analysis results and metrics.

Example usage:

```python
# Load data from CSV files
features_dict = load_features_from_csv(data_paths)

# Perform correlation analysis
results = analyze_multimodal_correlations(features_dict)
```
