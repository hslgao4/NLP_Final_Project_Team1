import pickle
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Amazon Books Reviews data from:
# https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews?select=books_data.csv


# Load the pickled model
with open('SVD_recommendations.pkl', 'rb') as f:
    lsa_model = pickle.load(f)


# Load the pickled X (data)
with open('X_matrix.pkl', 'rb') as f:
    X = pickle.load(f)


# Load the dataset
data = pd.read_parquet('/home/ubuntu/caitlin/1_DATS6312_NLP/Project/data/books_merged_clean.parquet')


# Function to preprocess the text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespaces
    text = " ".join(text.split())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    text = " ".join([word for word in tokens if word not in stop_words])
    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    text = " ".join([lemmatizer.lemmatize(word) for word in tokens])
    return text


# Function to clean Book title
def clean_title(title):
    # Lowercase all characters
    title = title.lower()
    # Remove comma and semicolon
    title = title.replace(',', '').replace(';', '')
    # Capitalize first word only
    title = title.capitalize()
    return title


# Function to generate recommendations
def get_recommendations(text_input, data, lsa_model, X):
    # Preprocess the text
    processed_text = preprocess_text(text_input)
    # Transform the preprocessed text input
    input_vector = lsa_model.transform([processed_text])
    # Compute cosine similarity between input vector and all book vectors
    similarities = cosine_similarity(input_vector, X)
    # Get indices of top recommendations
    top_indices = similarities.argsort()[0][::-1][:40]  # Get top 30 recommendations
    # Retrieve recommended books
    recommendations = data.iloc[top_indices]
    # Apply clean_title function to the 'Title' column
    recommendations = recommendations.copy() # ensure that any modifications are made to a separate copy
    recommendations.loc[:, 'Title'] = recommendations['Title'].apply(clean_title)
    # Convert 'review/summary' and 'Title' columns to lowercase for duplicate removal
    recommendations['review/summary_lower'] = recommendations['review/summary'].str.lower()
    recommendations['Title_lower'] = recommendations['Title'].str.lower()
    # Remove duplicates based on review summary
    recommendations = recommendations.drop_duplicates(subset=['review/summary_lower'], keep='first')
    # Remove duplicates based on Title
    recommendations = recommendations.drop_duplicates(subset=['Title_lower'], keep='first')
    # Sort recommendations by rating (review/score)
    recommendations = recommendations.sort_values(by='review/score', ascending=False)
    # Return top non-duplicate recommendations
    return recommendations.head(10)


# Main function
def main():
    # Take user input via terminal
    input_text = input("For Future Reads recommendations, type your recent five-star book review here:")
    # Make recommendations based on the input text
    recommendations = get_recommendations(input_text, data, lsa_model, X)
    # Print recommendations
    print(recommendations[['Title', 'categories', 'review/summary']])

if __name__ == "__main__":
    main()