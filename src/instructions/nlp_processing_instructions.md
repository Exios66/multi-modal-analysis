# NLP Processing Module Instructions

## Overview

The `nlp_processing.py` module provides natural language processing capabilities for analyzing textual data in multimodal analysis. It handles text preprocessing, feature extraction, sentiment analysis, and topic modeling using state-of-the-art NLP libraries.

## Features

- Text cleaning and preprocessing:
  - Lowercasing
  - Punctuation and number removal
  - Stopword removal
  - Lemmatization
  - Whitespace normalization
  - Error handling and validation

- Sentiment analysis:
  - Transformer-based sentiment classification
  - Batch processing for efficiency
  - Sentiment scores and labels
  - Memory-efficient processing
  - Error handling

- Topic modeling:
  - Latent Dirichlet Allocation (LDA)
  - Configurable number of topics
  - Document-topic distributions
  - Dominant topic identification
  - Dictionary filtering
  - Random seed for reproducibility

- Feature extraction:
  - TF-IDF vectorization
  - Configurable feature parameters
  - Document frequency thresholds
  - Sparse matrix handling
  - Feature naming
  - Memory-efficient processing

## Function Documentation

### clean_text(text: str) -> str

Cleans and preprocesses raw text data.

Parameters:

- text (str): Raw text input

Returns:

- str: Cleaned and preprocessed text

Raises:

- ValueError: If input text is None or empty
- TypeError: If input text is not a string

### preprocess_text_data(df: pd.DataFrame, text_column: str) -> pd.DataFrame

Applies text cleaning to a DataFrame column.

Parameters:

- df (pd.DataFrame): Input DataFrame
- text_column (str): Name of column containing text

Returns:

- pd.DataFrame: DataFrame with new 'cleaned_text' column

Raises:

- ValueError: If DataFrame empty or column missing

### extract_sentiment(df: pd.DataFrame, text_column: str = 'cleaned_text', batch_size: int = 32) -> pd.DataFrame

Extracts sentiment using transformers.

Parameters:

- df (pd.DataFrame): Input DataFrame
- text_column (str): Column with cleaned text
- batch_size (int): Processing batch size

Returns:

- pd.DataFrame: DataFrame with sentiment columns

Raises:

- ValueError: If DataFrame empty or column missing

### extract_tfidf_features(df: pd.DataFrame, text_column: str = 'cleaned_text', max_features: int = 1000, min_df: float = 0.01, max_df: float = 0.95) -> pd.DataFrame

Extracts TF-IDF features from text.

Parameters:

- df (pd.DataFrame): Input DataFrame
- text_column (str): Column with cleaned text
- max_features (int): Max number of features
- min_df (float): Min document frequency
- max_df (float): Max document frequency

Returns:

- pd.DataFrame: TF-IDF feature matrix

Raises:

- ValueError: If DataFrame empty or column missing

### perform_topic_modeling(df: pd.DataFrame, text_column: str = 'cleaned_text', num_topics: int = 5, passes: int = 10, random_state: int = 42) -> pd.DataFrame

Performs LDA topic modeling.

Parameters:

- df (pd.DataFrame): Input DataFrame
- text_column (str): Column with cleaned text
- num_topics (int): Number of topics
- passes (int): Training passes
- random_state (int): Random seed

Returns:

- pd.DataFrame: Topic distributions

Raises:

- ValueError: If DataFrame empty or column missing
