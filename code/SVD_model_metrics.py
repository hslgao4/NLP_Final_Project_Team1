# Import
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import matplotlib.pyplot as plt


# Function to load and clean the data
def load_and_clean_dataset(file_path, summary_column='review/summary', text_column='review/text'):
    # Load the dataset
    data = pd.read_parquet(file_path)
    # Fill missing values for "review/title" and "review/summary" columns with empty strings
    data.fillna({'review/title': '', 'review/summary': ''}, inplace=True)
    # Combine text data
    data['review_combined'] = data[summary_column] + ' ' + data[text_column]
    # Drop rows with missing values for specified columns
    columns_to_check = ['review_combined']
    cleaned_data = data.dropna(subset=columns_to_check, how='any')
    return cleaned_data


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


# Function to create a pipeline combining TF-IDF vectorization and SVD
def train_lsa_model(data, combined_column='review_combined', max_df=0.5, min_df=2, n_components=100, random_state=42):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words='english', use_idf=True)
    # Create TruncatedSVD model
    svd_model = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=random_state)
    # Create pipeline
    lsa_model = make_pipeline(vectorizer, svd_model)
    # Fit and transform the data
    X = lsa_model.fit_transform(data[combined_column])
    # Save the X matrix
    with open('X_matrix.pkl', 'wb') as f:
        pickle.dump(X, f)
    return X, lsa_model


# Function to generate recommendations
def get_recommendations(text_input, data, lsa_model, X):
    # Preprocess the text
    processed_text = preprocess_text(text_input)
    # Transform the preprocessed text input
    input_vector = lsa_model.transform([processed_text])
    # Compute cosine similarity between input vector and all book vectors
    similarities = cosine_similarity(input_vector, X)
    # Get indices of top recommendations
    top_indices = similarities.argsort()[0][::-1][:10]  # Get top 10 recommendations
    # Retrieve recommended books
    recommendations = data.iloc[top_indices]
    return recommendations


# Function to save the trained model
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


# Function to find optimal number of components using the elbow method
def find_optimal_components(X):
    explained_variances = []
    num_components_range = range(1, min(X.shape)+1)
    for n_components in num_components_range:
        svd_model = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=42)
        svd_model.fit(X)
        explained_variance = np.sum(svd_model.explained_variance_ratio_)
        explained_variances.append(explained_variance)
    return explained_variances


# Function to request user input, call the SVD model, and output book recommendations
def main():
    # Load data
    file_path = "/home/ubuntu/caitlin/NLP_Project_Team1/data/books_merged_clean.parquet"
    data = load_and_clean_dataset(file_path)
    # call model
    X, lsa_model = train_lsa_model(data, combined_column='review_combined')
    # Find optimal number of components
    explained_variances = find_optimal_components(X)
    # Plotting the elbow method
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variances) + 1), explained_variances, marker='o', linestyle='-')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance')
    plt.title('Elbow Method for Optimal Number of Components')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()