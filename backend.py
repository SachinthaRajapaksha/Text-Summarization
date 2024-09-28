from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
import PyPDF2
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# Load your fine-tuned summarization model
summarizer = pipeline('summarization', model='finetuned_summarizer')

# Adjust the summarization logic to support different summary lengths
def summarize_text(text, summary_type):
    # Calculate the token length of the input text
    length = len(text.split())  # You could also use tokenizer.encode(text) if needed

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
    summary = summarizer(text, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4,early_stopping=True)
    
    return summary[0]['summary_text']

# Load the keyword extraction model and tokenizer
model_name = "agentlans/flan-t5-small-keywords"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def extract_keywords(text, max_length=512):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    # Generate keywords using the pre-trained model
    outputs = model.generate(**inputs, max_length=max_length)

    # Decode the generated output back into text (keywords)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Split keywords and remove duplicates
    keywords = list(set(decoded_output.split('||')))
    
    return keywords

# Read PDF data and summarize
def process_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Preprocess text by tokenizing and removing stopwords
def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())
    return ' '.join([word for word in words if word.isalnum() and word not in stop_words])

# Generate topics using LDA
def extract_topics(text, num_topics=3, num_words=5):
    # Preprocess the text
    processed_text = preprocess_text(text)

    # Use TF-IDF to vectorize the text
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([processed_text])

    # Apply LDA for topic modeling
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    # Get the topics with the most significant words
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        topic_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append(' '.join(topic_words))

    return topics