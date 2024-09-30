from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import PyPDF2
import nltk


# Read PDF data and summarize
def process_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Load your fine-tuned summarization model
summarizer = pipeline('summarization', model='finetuned_summarizer')

def summarize_text(text, summary_type):
    
    length = len(text.split())

    if summary_type == 'short':
        max_length = int(length / 8)
        min_length = int(length / 10)

    elif summary_type == 'medium':
        max_length = int(length / 4)
        min_length = int(length / 5)

    elif summary_type == 'detailed':
        max_length = int(length / 2)
        min_length = int(length / 3)

    else:
        raise ValueError("Invalid summary type. Choose from 'short', 'medium', or 'detailed'.")

    # Generate the summary
    summary = summarizer(text, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)

    return summary[0]['summary_text']

#the keyword extraction
nltk.download('punkt')
nltk.download('stopwords')

def extract_keywords(text, ngram_range=(1, 3), top_n=20):
    # Tokenize the text and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    
    # Join tokens into a single string for TfidfVectorizer
    text = ' '.join(tokens)

    # Create a TfidfVectorizer to generate n-grams
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=top_n)

    # Fit and transform the text using TF-IDF
    tfidf_matrix = vectorizer.fit_transform([text])  # Pass as a list

    # Get the TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    tfidf_scores = dict(zip(feature_names, scores))

    # Sort the keywords by highest TF-IDF score
    sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)

    # Extract the top N keywords
    top_keywords = [keyword for keyword, score in sorted_keywords[:top_n]]

    return top_keywords



# topic modelling using our model

model_name_topics = "topic_model_t5"
topic_model = pipeline("text2text-generation", model=model_name_topics)

def generate_topics(text):
    
    generated_topics = topic_model(text, max_length=50, num_return_sequences=5, do_sample=True, top_k=50, top_p=0.95)

    topics = [topic['generated_text'] for topic in generated_topics]
    
    return topics



# Sentiment analysis

# Initialize the VADER sentiment intensity analyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    # Get the sentiment scores using VADER
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']

    # Determine sentiment based on compound score
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def sentiment_summary(article):
    sentiment = analyze_sentiment(article)
    
    if sentiment == 'positive':
        summary = "Document you entered reflects an overall 'Positive' narrative about the highlighted subject."
    elif sentiment == 'negative':
        summary = "Document you entered reflects an overall 'Negative' narrative about the highlighted subject."
    else:
        summary = "Document you entered reflects an overall 'Neutral' narrative about the highlighted subject."

    return summary
