import pandas as pd import numpy as np import logging import re import nltk from nltk.corpus import stopwords from nltk.stem import WordNetLemmatizer import spacy from gensim import corpora, models from sklearn.feature_extraction.text import TfidfVectorizer from transformers import pipeline

Configure logging

logging.basicConfig(level=logging.INFO) logger = logging.getLogger(name)

Initialize spaCy model

nlp = spacy.load('en_core_web_sm')

Initialize NLTK tools

lemmatizer = WordNetLemmatizer() stop_words = set(stopwords.words('english'))

def clean_text(text): """ Clean and preprocess text data.

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
def preprocess_text_data(df, text_column): """ Apply text cleaning to a specific column in the DataFrame.

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
def extract_sentiment(df, text_column='cleaned_text'): """ Extract sentiment scores from text data using Hugging Face transformers.

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
def extract_tfidf_features(df, text_column='cleaned_text', max_features=1000): """ Extract TF-IDF features from text data.

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
def perform_topic_modeling(df, text_column='cleaned_text', num_topics=5): """ Perform topic modeling using Latent Dirichlet Allocation (LDA).

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
