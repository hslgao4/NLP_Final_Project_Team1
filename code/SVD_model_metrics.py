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
import seaborn as sns
from gensim import models
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

# Amazon Books Reviews data from:
# https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews?select=books_data.csv


# Function to load and clean the data
def load_and_clean_dataset(file_path, summary_column='review/summary', text_column='review/text'):
    # Load the dataset from Parquet file
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
def train_lsa_model(data, combined_column='review_combined', max_df=0.5, min_df=2, n_components=17, random_state=42):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words='english', use_idf=True)
    # Create TruncatedSVD model
    svd_model = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=random_state)
    # Create pipeline
    lsa_model = make_pipeline(vectorizer, svd_model)
    # Fit and transform the data
    X = lsa_model.fit_transform(data[combined_column])
    # Save the X matrix
    with open('X_matrix_17.pkl', 'wb') as f:
        pickle.dump(X, f)
    return X, lsa_model


# Function to save the trained model
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


# Function to find both cumulative explained variance and explained variance ratio for different numbers of components
def find_variance_metrics(X, max_components=None, variance_threshold=0.9):
    cumulative_variances = []
    explained_variances = []

    if max_components is None and variance_threshold is None:
        raise ValueError("Either 'max_components' or 'variance_threshold' must be specified.")

    svd_model = TruncatedSVD(n_components=X.shape[1], algorithm='randomized', random_state=42)
    svd_model.fit(X)

    if variance_threshold:
        total_variance = np.sum(svd_model.explained_variance_ratio_)
        variances = svd_model.explained_variance_ratio_
        cumulative_variance = 0
        for i, variance in enumerate(variances):
            cumulative_variance += variance
            if cumulative_variance >= variance_threshold:
                max_components = i + 1
                break

    num_components_range = range(1, min(X.shape) + 1) if max_components is None else range(1, max_components + 1)

    for n_components in num_components_range:
        svd_model = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=42)
        svd_model.fit(X)
        cumulative_variance = np.sum(svd_model.explained_variance_ratio_)
        explained_variance_ratio = svd_model.explained_variance_ratio_
        cumulative_variances.append(cumulative_variance)
        explained_variances.append(explained_variance_ratio)

    return cumulative_variances, explained_variances


def plot_cumulative_variance(cumulative_variances, max_components):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_components + 1), cumulative_variances[:max_components], marker='', linestyle='-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance for Different Numbers of Components')
    plt.grid(True)
    plt.show()


def plot_explained_variance_ratio(explained_variances):
    plt.figure(figsize=(8, 6))
    for i, var_ratio in enumerate(explained_variances):
        plt.plot(range(1, len(var_ratio) + 1), var_ratio)
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio for Different Numbers of Components')
    plt.grid(True)
    plt.xticks(np.arange(0, 20, 2))
    plt.show()


# Function to compute coherence score
def compute_coherence_score(X, tokenized_documents, dictionary):
    lsa_model = models.LsiModel(X, id2word=dictionary, num_topics=17)
    coherence_model = CoherenceModel(model=lsa_model, texts=tokenized_documents, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    return coherence_score


# Function to call the SVD model and output model metrics
def main():
    # Load saved X matrix if skipping training code
    #with open('X_matrix_17.pkl', 'rb') as f:
    #    X = pickle.load(f)
    # Load data
    file_path = "data/books_merged_clean.parquet" # add path to parquet file
    data = load_and_clean_dataset(file_path)
    # call model
    X, lsa_model = train_lsa_model(data, combined_column='review_combined')
    # save the model
    save_model(lsa_model, "SVD_recommendations_17.pkl")
    # Find both cumulative explained variance and explained variance ratio
    cumulative_variances, explained_variances = find_variance_metrics(X)
    # Plot both metrics
    plot_cumulative_variance(cumulative_variances, max_components=17)
    plot_explained_variance_ratio(explained_variances)



if __name__ == "__main__":
    main()