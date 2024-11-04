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
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    logger.warning("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Initialize NLTK tools
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.warning("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Clean and preprocess text data.

    Steps:
    - Lowercasing
    - Removing punctuation and numbers
    - Removing stopwords
    - Lemmatization

    Args:
        text (str): Raw text input

    Returns:
        str: Cleaned and preprocessed text

    Raises:
        ValueError: If input text is None or empty
        TypeError: If input text is not a string
    """
    if text is None or not isinstance(text, str):
        raise TypeError("Input text must be a string")
    if not text.strip():
        raise ValueError("Input text cannot be empty")

    try:
        # Lowercase
        text = text.lower()
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        
        if not tokens:
            logger.warning("No tokens remained after cleaning")
            return ""
            
        cleaned_text = ' '.join(tokens)
        return cleaned_text

    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        raise

def preprocess_text_data(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Apply text cleaning to a specific column in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the column containing text data

    Returns:
        pd.DataFrame: DataFrame with a new column 'cleaned_text'

    Raises:
        ValueError: If DataFrame is empty or text_column doesn't exist
    """
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty")
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    try:
        df['cleaned_text'] = df[text_column].astype(str).apply(clean_text)
        logger.info(f"Text data in column '{text_column}' cleaned and added as 'cleaned_text'")
        return df

    except Exception as e:
        logger.error(f"Error preprocessing text data: {str(e)}")
        raise

def extract_sentiment(df: pd.DataFrame, text_column: str = 'cleaned_text', batch_size: int = 32) -> pd.DataFrame:
    """Extract sentiment scores from text data using Hugging Face transformers.

    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the column containing cleaned text data
        batch_size (int): Batch size for sentiment analysis

    Returns:
        pd.DataFrame: DataFrame with a new column 'sentiment'

    Raises:
        ValueError: If DataFrame is empty or text_column doesn't exist
    """
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty")
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    try:
        sentiment_pipeline = pipeline("sentiment-analysis", batch_size=batch_size)
        texts = df[text_column].tolist()
        
        # Process in batches to handle memory efficiently
        sentiments = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_sentiments = sentiment_pipeline(batch)
            sentiments.extend(batch_sentiments)

        df['sentiment'] = [sentiment['label'] for sentiment in sentiments]
        df['sentiment_score'] = [sentiment['score'] for sentiment in sentiments]
        
        logger.info("Sentiment analysis completed and added as 'sentiment' and 'sentiment_score'")
        return df

    except Exception as e:
        logger.error(f"Error extracting sentiment: {str(e)}")
        raise

def extract_tfidf_features(
    df: pd.DataFrame,
    text_column: str = 'cleaned_text',
    max_features: int = 1000,
    min_df: float = 0.01,
    max_df: float = 0.95
) -> pd.DataFrame:
    """Extract TF-IDF features from text data.

    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the column containing cleaned text data
        max_features (int): Maximum number of features for TF-IDF
        min_df (float): Minimum document frequency threshold
        max_df (float): Maximum document frequency threshold

    Returns:
        pd.DataFrame: DataFrame containing TF-IDF features

    Raises:
        ValueError: If DataFrame is empty or text_column doesn't exist
    """
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty")
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        
        tfidf_matrix = vectorizer.fit_transform(df[text_column])
        feature_names = vectorizer.get_feature_names_out()
        
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names,
            index=df.index
        )
        
        logger.info(f"TF-IDF features extracted: {len(feature_names)} features")
        return tfidf_df

    except Exception as e:
        logger.error(f"Error extracting TF-IDF features: {str(e)}")
        raise

def perform_topic_modeling(
    df: pd.DataFrame,
    text_column: str = 'cleaned_text',
    num_topics: int = 5,
    passes: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """Perform topic modeling using Latent Dirichlet Allocation (LDA).

    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the column containing cleaned text data
        num_topics (int): Number of topics to extract
        passes (int): Number of passes through the corpus during training
        random_state (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: DataFrame containing topic distributions for each document

    Raises:
        ValueError: If DataFrame is empty or text_column doesn't exist
    """
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty")
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    try:
        # Tokenize and create dictionary
        texts = df[text_column].apply(lambda x: x.split()).tolist()
        dictionary = corpora.Dictionary(texts)
        
        # Filter extreme frequencies
        dictionary.filter_extremes(no_below=2, no_above=0.9)
        
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        # Build LDA model
        lda_model = models.LdaModel(
            corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=passes,
            random_state=random_state
        )
        
        # Get topic distributions
        topics = lda_model.get_document_topics(corpus, minimum_probability=0.0)
        
        # Convert to DataFrame
        topic_probs = np.zeros((len(texts), num_topics))
        for doc_idx, doc_topics in enumerate(topics):
            for topic_idx, prob in doc_topics:
                topic_probs[doc_idx, topic_idx] = prob
        
        topics_df = pd.DataFrame(
            topic_probs,
            columns=[f"topic_{i}" for i in range(num_topics)],
            index=df.index
        )
        
        # Add dominant topic for each document
        topics_df['dominant_topic'] = topics_df.idxmax(axis=1)
        
        logger.info(f"Topic modeling completed: {num_topics} topics extracted")
        return topics_df

    except Exception as e:
        logger.error(f"Error performing topic modeling: {str(e)}")
        raise
