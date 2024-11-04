# Comprehensive Python Application for Psychometric and Neurological Data Analysis with NLP and ML Integration

## Table of Contents

1. Introduction
2. Project Overview
3. System Architecture
4. Implementation Steps
    • Step 1: Set Up the Development Environment
    • Step 2: Organize the Project Directory
    • Step 3: Develop the Data Ingestion Module
    • Step 4: Develop the Data Preprocessing Module
    • Step 5: Develop the Feature Extraction Module
    • Step 6: Develop the Correlation Analysis Module
    • Step 7: Develop the Visualization Module
    • Step 8: Develop the NLP Processing Module
    • Step 9: Develop the Machine Learning Module
    • Step 10: Update the Main Application Workflow
    • Step 11: Deployment of ML Models
5. Code Modules
6. Dependencies and Installation
7. Running the Application
8. Conclusion
9. Appendices
    • Appendix A: Full Source Code
    • Appendix B: Dataset Format Examples
    • Appendix C: Sample .csv Files

### Introduction

Overview

This expanded guide outlines the development of a Python application designed for comprehensive analysis of psychometric, vitals, neuroimaging, and survey data. By integrating Natural Language Processing (NLP) and Machine Learning (ML) algorithms, the application will not only preprocess and visualize data but also perform advanced analyses to uncover deeper insights, predict outcomes, and assist in decision-making processes.

Key Enhancements

 • Natural Language Processing (NLP): Analyze textual survey responses to extract sentiments, topics, and other linguistic features.
 • Machine Learning (ML): Implement predictive models to forecast health outcomes, classify participant responses, and identify patterns within the data.
 • Advanced Correlation and Statistical Analysis: Utilize ML-driven statistical methods to assess relationships within the data.
 • Deployment: Deploy ML models as APIs for real-time inference and integration with other systems.

### Project Overview

#### Core Functionalities

1. Data Ingestion: Efficiently reads data from multiple formats and validates their integrity.
2. Data Preprocessing: Cleans, normalizes, and synchronizes data to prepare it for analysis.
3. Feature Extraction: Derives meaningful metrics and features from raw data sources, including textual data.
4. Natural Language Processing (NLP): Processes and extracts features from textual survey responses.
5. Machine Learning (ML): Applies ML algorithms for predictive analytics and pattern recognition.
6. Correlation and Statistical Analysis: Identifies and quantifies relationships between different datasets.
7. Visualization: Creates intuitive visual representations of data and analytical findings.
8. Deployment: Provides APIs for ML model deployment, enabling real-time data analysis and prediction.

### System Architecture

#### Modular Design

The enhanced system architecture incorporates additional layers and modules to support NLP and ML functionalities while maintaining separation of concerns, ease of maintenance, and scalability.

 • Input Layer:
 • Data Ingestion Module: Handles reading data from various sources and formats with robust error handling and logging.
 • Processing Layer:
 • Data Preprocessing Module: Cleans and formats the data, including handling missing values, outlier removal, and artifact correction.
 • Feature Extraction Module: Extracts key features from the preprocessed data, including statistical features and NLP-derived features.
 • Analysis Layer:
 • Correlation Analysis Module: Performs statistical analyses to find correlations using multiple methods.
 • Machine Learning Module: Implements ML algorithms for predictive modeling, classification, and clustering.
 • Output Layer:
 • Visualization Module: Generates visual insights such as plots and heatmaps.
 • Deployment Module: Deploys trained ML models as APIs for real-time inference.

This modular architecture ensures that each component can be developed, tested, and maintained independently, enhancing the overall robustness and scalability of the application.

### Implementation Steps

#### Step 1: Set Up the Development Environment

Detailed Steps

 1. Install Python:
 • Ensure Python 3.8 or higher is installed. You can download it from the official website.
 2. Create a Virtual Environment:
 • Virtual environments help manage dependencies and avoid conflicts.

python3 -m venv venv

#### Step 2: Activate the Virtual Environment

 • On macOS/Linux:

source venv/bin/activate

 • On Windows:

venv\Scripts\activate

#### Step 3: Install Required Packages

 • It's good practice to pin package versions for reproducibility.

pip install numpy pandas scipy scikit-learn matplotlib seaborn mne nltk spacy gensim transformers flask fastapi uvicorn joblib

 • Alternatively, use the requirements.txt for installation:

pip install -r requirements.txt

Updated requirements.txt

numpy==1.24.3
pandas==1.5.3
scipy==1.11.2
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
mne==1.3.1
nltk==3.8.1
spacy==3.6.0
gensim==4.3.1
transformers==4.30.0
flask==2.3.2
fastapi==0.95.1
uvicorn==0.22.0
joblib==1.3.2

Note: Depending on your specific NLP and ML needs, you might need to install additional packages or specific versions. Ensure compatibility among package versions to avoid conflicts.

#### Step 4: Download NLTK Data

 • Some NLTK functionalities require downloading additional data packages.

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#### Step 5: Download spaCy Models

 • For advanced NLP tasks, spaCy models need to be downloaded.

python -m spacy download en_core_web_sm

#### Step 6: Organize the Project Directory

Recommended Directory Structure

    project_root/
    │
    ├── data/
    │   ├── eye_tracking/
    │   ├── face_heatmap/
    │   ├── vitals/
    │   ├── eeg/
    │   └── surveys/
    │
    ├── src/
    │   ├── __init__.py
    │   ├── data_ingestion.py
    │   ├── data_preprocessing.py
    │   ├── feature_extraction.py
    │   ├── correlation_analysis.py
    │   ├── visualization.py
    │   ├── nlp_processing.py
    │   └── machine_learning.py
    │
    ├── models/
    │   └── trained_models/
    │
    ├── deployment/
    │   ├── api.py
    │   └── requirements.txt
    │
    ├── tests/
    │   ├── __init__.py
    │   ├── test_data_ingestion.py
    │   ├── test_data_preprocessing.py
    │   ├── test_feature_extraction.py
    │   ├── test_correlation_analysis.py
    │   ├── test_visualization.py
    │   ├── test_nlp_processing.py
    │   └── test_machine_learning.py
    │
    ├── output/
    │   └── plots/
    │
    ├── main.py
    ├── requirements.txt
    ├── README.md
    └── .gitignore

#### Additional Considerations

 • models/ Directory: Stores trained ML models for deployment and future use.
 • deployment/ Directory: Contains scripts and configurations for deploying ML models as APIs.
 • nlp_processing.py: Handles NLP-specific preprocessing and feature extraction.
 • machine_learning.py: Contains ML algorithms for training, evaluation, and prediction.
 • test_nlp_processing.py and test_machine_learning.py: Unit tests for the new modules.

#### Sample .gitignore Update

#### Trained ML models

models/trained_models/

#### Deployment

deployment/__pycache__/
deployment/*.pyc

#### Other additions

output/

#### Step 7: Develop the Data Ingestion Module

Refer to the previous assistant message for the data_ingestion.py module. Ensure that it remains robust to handle any additional data types introduced by NLP and ML processes.

#### Step 8: Develop the Data Preprocessing Module

Refer to the previous assistant message for the data_preprocessing.py module. Ensure that it remains robust and includes any additional preprocessing steps required for ML and NLP tasks.

#### Step 9: Develop the Feature Extraction Module

Refer to the previous assistant message for the feature_extraction.py module. Ensure that it can handle additional features derived from NLP processes.

#### Step 10: Develop the Correlation Analysis Module

Refer to the previous assistant message for the correlation_analysis.py module. Ensure that it can support additional statistical analyses driven by ML requirements.

#### Step 11: Develop the Visualization Module

Refer to the previous assistant message for the visualization.py module. Ensure that it can handle visualizations related to ML model performance and NLP analysis.

#### Step 12: Develop the NLP Processing Module

#### Purpose

To process and extract features from textual survey responses, enabling sentiment analysis, topic modeling, and other linguistic feature extractions essential for comprehensive data analysis.

File: src/nlp_processing.py

import pandas as pd
import numpy as np
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

#### Configure ML Module Logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#### Initialize spaCy model

nlp = spacy.load('en_core_web_sm')

#### Initialize NLTK tools

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean and preprocess text data.

    Steps:
    - Lowercasing
    - Removing punctuation and numbers
    - Removing stopwords
    - Lemmatization
    
    Parameters:
    - text: string
    
    Returns:
    - cleaned_text: string
    """
    try:
        # Lowercase
        text = text.lower()
        # Remove punctuation and numbers
        text = re.sub(r'[^a-z\s]', '', text)
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        cleaned_text = ' '.join(tokens)
        return cleaned_text
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        raise

def preprocess_text_data(df, text_column):
    """
    Apply text cleaning to a specific column in the DataFrame.

    Parameters:
    - df: pandas DataFrame
    - text_column: string, name of the column containing text data
    
    Returns:
    - df: pandas DataFrame with a new column 'cleaned_text'
    """
    try:
        df['cleaned_text'] = df[text_column].apply(clean_text)
        logger.info(f"Text data in column '{text_column}' cleaned and added as 'cleaned_text'")
        return df
    except Exception as e:
        logger.error(f"Error preprocessing text data: {e}")
        raise

def extract_sentiment(df, text_column='cleaned_text'):
    """
    Extract sentiment scores from text data using Hugging Face transformers.

    Parameters:
    - df: pandas DataFrame
    - text_column: string, name of the column containing cleaned text data
    
    Returns:
    - df: pandas DataFrame with a new column 'sentiment'
    """
    try:
        sentiment_pipeline = pipeline("sentiment-analysis")
        sentiments = sentiment_pipeline(df[text_column].tolist())
        df['sentiment'] = [sentiment['label'] for sentiment in sentiments]
        logger.info("Sentiment analysis completed and added as 'sentiment'")
        return df
    except Exception as e:
        logger.error(f"Error extracting sentiment: {e}")
        raise

def extract_tfidf_features(df, text_column='cleaned_text', max_features=1000):
    """
    Extract TF-IDF features from text data.

    Parameters:
    - df: pandas DataFrame
    - text_column: string, name of the column containing cleaned text data
    - max_features: int, maximum number of features for TF-IDF
    
    Returns:
    - tfidf_df: pandas DataFrame containing TF-IDF features
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(df[text_column])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        logger.info("TF-IDF features extracted")
        return tfidf_df
    except Exception as e:
        logger.error(f"Error extracting TF-IDF features: {e}")
        raise

def perform_topic_modeling(df, text_column='cleaned_text', num_topics=5):
    """
    Perform topic modeling using Latent Dirichlet Allocation (LDA).

    Parameters:
    - df: pandas DataFrame
    - text_column: string, name of the column containing cleaned text data
    - num_topics: int, number of topics to extract
    
    Returns:
    - topics_df: pandas DataFrame containing topic distributions for each document
    """
    try:
        # Tokenize and create dictionary
        texts = df[text_column].apply(lambda x: x.split()).tolist()
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        # Build LDA model
        lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
        topics = lda_model.get_document_topics(corpus)
        
        # Convert to DataFrame
        topics_df = pd.DataFrame([[topic_prob for _, topic_prob in doc] for doc in topics], columns=[f"topic_{i}" for i in range(num_topics)])
        logger.info("Topic modeling completed and topic distributions extracted")
        return topics_df
    except Exception as e:
        logger.error(f"Error performing topic modeling: {e}")
        raise

#### Enhancements

 • Text Cleaning: Implements comprehensive text cleaning, including lowercasing, removal of punctuation and numbers, stopword removal, and lemmatization.
 • Sentiment Analysis: Utilizes Hugging Face's transformers pipeline to extract sentiment labels (e.g., POSITIVE, NEGATIVE) from cleaned text data.
 • TF-IDF Feature Extraction: Converts textual data into numerical features using TF-IDF vectorization, facilitating ML model training.
 • Topic Modeling: Applies LDA to identify underlying topics within textual data, adding topic distributions as features.

Step 9: Develop the Machine Learning Module

Purpose

To implement ML algorithms for predictive analytics, classification, clustering, and regression tasks based on the extracted features. This module will handle model training, evaluation, hyperparameter tuning, and saving trained models for deployment.

File: src/machine_learning.py

import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#### Configure Logging for Machine Learning Module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_classification_model(X, y, model_type='RandomForest', params=None):
    """
    Train a classification model.

    Parameters:
    - X: pandas DataFrame, feature matrix
    - y: pandas Series or array-like, target labels
    - model_type: string, type of model ('LogisticRegression', 'RandomForest')
    - params: dict, hyperparameters for GridSearchCV
    
    Returns:
    - best_model: trained model with best parameters
    """
    try:
        if model_type == 'LogisticRegression':
            model = LogisticRegression(max_iter=1000)
        elif model_type == 'RandomForest':
            model = RandomForestClassifier()
        else:
            raise ValueError("Unsupported model type")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        if params:
            grid = GridSearchCV(pipeline, params, cv=5, n_jobs=-1, scoring='accuracy')
            grid.fit(X, y)
            best_model = grid.best_estimator_
            logger.info(f"{model_type} trained with best parameters: {grid.best_params_}")
        else:
            pipeline.fit(X, y)
            best_model = pipeline
            logger.info(f"{model_type} trained without hyperparameter tuning")
        
        return best_model
    except Exception as e:
        logger.error(f"Error training classification model: {e}")
        raise

def train_regression_model(X, y, model_type='RandomForest', params=None):
    """
    Train a regression model.

    Parameters:
    - X: pandas DataFrame, feature matrix
    - y: pandas Series or array-like, target variable
    - model_type: string, type of model ('LinearRegression', 'RandomForest')
    - params: dict, hyperparameters for GridSearchCV
    
    Returns:
    - best_model: trained model with best parameters
    """
    try:
        if model_type == 'LinearRegression':
            model = LinearRegression()
        elif model_type == 'RandomForest':
            model = RandomForestRegressor()
        else:
            raise ValueError("Unsupported model type")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        
        if params:
            grid = GridSearchCV(pipeline, params, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
            grid.fit(X, y)
            best_model = grid.best_estimator_
            logger.info(f"{model_type} trained with best parameters: {grid.best_params_}")
        else:
            pipeline.fit(X, y)
            best_model = pipeline
            logger.info(f"{model_type} trained without hyperparameter tuning")
        
        return best_model
    except Exception as e:
        logger.error(f"Error training regression model: {e}")
        raise

def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate a classification model.

    Parameters:
    - model: trained classification model
    - X_test: pandas DataFrame, test feature matrix
    - y_test: pandas Series or array-like, true labels
    
    Returns:
    - report: string, classification report
    - conf_matrix: pandas DataFrame, confusion matrix
    """
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred))
        logger.info("Classification model evaluated")
        return report, conf_matrix
    except Exception as e:
        logger.error(f"Error evaluating classification model: {e}")
        raise

def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluate a regression model.

    Parameters:
    - model: trained regression model
    - X_test: pandas DataFrame, test feature matrix
    - y_test: pandas Series or array-like, true target values
    
    Returns:
    - mse: float, Mean Squared Error
    - r2: float, R-squared score
    """
    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info("Regression model evaluated")
        return mse, r2
    except Exception as e:
        logger.error(f"Error evaluating regression model: {e}")
        raise

def train_clustering_model(X, n_clusters=5, random_state=42):
    """
    Train a clustering model using K-Means.

    Parameters:
    - X: pandas DataFrame, feature matrix
    - n_clusters: int, number of clusters
    - random_state: int, random seed
    
    Returns:
    - model: trained KMeans model
    - labels: array-like, cluster labels for each sample
    """
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
        model.fit(X_scaled)
        labels = model.labels_
        logger.info(f"K-Means clustering performed with {n_clusters} clusters")
        return model, labels
    except Exception as e:
        logger.error(f"Error training clustering model: {e}")
        raise

def save_model(model, filepath):
    """
    Save a trained model to disk.

    Parameters:
    - model: trained model
    - filepath: string, path to save the model
    """
    try:
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def load_model(filepath):
    """
    Load a trained model from disk.

    Parameters:
    - filepath: string, path to the saved model
    
    Returns:
    - model: loaded model
    """
    try:
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

Enhancements

 • Classification Models: Implements Logistic Regression and Random Forest Classifier with options for hyperparameter tuning via GridSearchCV.
 • Regression Models: Implements Linear Regression and Random Forest Regressor with options for hyperparameter tuning.
 • Clustering Models: Implements K-Means clustering with feature scaling.
 • Model Evaluation: Provides functions to evaluate classification and regression models using appropriate metrics.
 • Model Persistence: Includes functions to save and load trained models using joblib.
 • Pipeline Integration: Utilizes Scikit-learn pipelines for streamlined preprocessing and modeling.

Step 10: Update the Main Application Workflow

Purpose

To integrate the newly developed NLP and ML modules into the existing workflow, ensuring seamless data flow from ingestion to deployment. This includes handling textual survey data, extracting features, training ML models, evaluating performance, and deploying models for real-time inference.

Updated File: main.py

import os
import pandas as pd
import logging
import joblib

#### Import custom modules

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
from src.nlp_processing import (
    preprocess_text_data,
    extract_sentiment,
    extract_tfidf_features,
    perform_topic_modeling
)
from src.machine_learning import (
    train_classification_model,
    train_regression_model,
    evaluate_classification_model,
    evaluate_regression_model,
    train_clustering_model,
    save_model
)

#### Configure logging

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
        
        # NLP Processing on Survey Data
        # Assuming survey_data has a 'response_text' column for textual responses
        if 'response_text' in survey_data.columns:
            survey_data = preprocess_text_data(survey_data, text_column='response_text')
            survey_data = extract_sentiment(survey_data, text_column='cleaned_text')
            tfidf_features = extract_tfidf_features(survey_data, text_column='cleaned_text', max_features=500)
            topic_features = perform_topic_modeling(survey_data, text_column='cleaned_text', num_topics=5)
            # Combine NLP features with survey data
            survey_features_nlp = pd.concat([tfidf_features, topic_features], axis=1)
        else:
            logger.warning("No 'response_text' column found in survey data. Skipping NLP processing.")
            survey_features_nlp = pd.DataFrame()
        
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
        
        # Incorporate NLP Features if available
        if not survey_features_nlp.empty:
            features_df = pd.concat([features_df, survey_features_nlp], axis=1)
            logger.info(f"NLP features added to features dataframe. New shape: {features_df.shape}")
        
        # Correlation Analysis
        corr_matrix = compute_correlations(features_df, method='pearson')
        significant_corrs = identify_significant_correlations(corr_matrix, threshold=0.5, method='pearson')
        logger.info(f"Significant correlations:\n{significant_corrs}")
        
        # Regression Analysis (Example: Predicting avg_heart_rate)
        if 'avg_heart_rate' in features_df.columns:
            # Define target and features
            y = features_df['avg_heart_rate']
            X = features_df.drop(columns=['avg_heart_rate'])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Regression Model
            params = {
                'regressor__n_estimators': [100, 200],
                'regressor__max_depth': [None, 10, 20]
            }
            reg_model = train_regression_model(X_train, y_train, model_type='RandomForest', params=params)
            
            # Evaluate Regression Model
            mse, r2 = evaluate_regression_model(reg_model, X_test, y_test)
            logger.info(f"Regression Model - MSE: {mse}, R2: {r2}")
            
            # Save Regression Model
            save_model(reg_model, 'models/trained_models/regression_model.joblib')
        else:
            logger.warning("Target feature 'avg_heart_rate' not found in features. Skipping regression analysis.")
            reg_model = None
        
        # Classification Analysis (Example: Sentiment Classification)
        if 'sentiment' in survey_data.columns and not survey_features_nlp.empty:
            # Encode sentiment labels
            y_class = survey_data['sentiment'].apply(lambda x: 1 if x == 'POSITIVE' else 0)
            X_class = survey_features_nlp
            
            # Split data
            X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
            
            # Train Classification Model
            params_clf = {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20]
            }
            clf_model = train_classification_model(X_train_clf, y_train_clf, model_type='RandomForest', params=params_clf)
            
            # Evaluate Classification Model
            clf_report, clf_conf_matrix = evaluate_classification_model(clf_model, X_test_clf, y_test_clf)
            logger.info(f"Classification Report:\n{clf_report}")
            logger.info(f"Confusion Matrix:\n{clf_conf_matrix}")
            
            # Save Classification Model
            save_model(clf_model, 'models/trained_models/classification_model.joblib')
        else:
            logger.warning("Sentiment data not available or NLP features missing. Skipping classification analysis.")
            clf_model = None
        
        # Clustering Analysis (Example: K-Means Clustering on Combined Features)
        clustering_features = features_df.copy()
        clustering_model, labels = train_clustering_model(clustering_features, n_clusters=3)
        features_df['cluster'] = labels
        logger.info("Clustering analysis completed and cluster labels added to features dataframe.")
        
        # Save Clustering Model
        save_model(clustering_model, 'models/trained_models/clustering_model.joblib')
        
        # Visualization
        # Ensure output directories exist
        output_dir = 'output/plots'
        os.makedirs(output_dir, exist_ok=True)
        
        # Correlation Heatmap
        plot_correlation_heatmap(corr_matrix, output_path=os.path.join(output_dir, 'correlation_heatmap.png'))
        
        # Feature Distributions
        plot_feature_distributions(features_df, output_path=os.path.join(output_dir, 'feature_distributions.png'))
        
        # Regression Results
        if reg_model:
            plot_regression_results(reg_model, X_test, y_test, output_path=os.path.join(output_dir, 'regression_results.png'))
