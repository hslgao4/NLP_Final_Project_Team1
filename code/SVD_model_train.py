import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
import pickle

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
    with open('X_matrix2.pkl', 'wb') as f:
        pickle.dump(X, f)
    return X, lsa_model


# Function to save the trained model
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def main():
    # Load data
    file_path = "/home/ubuntu/caitlin/NLP_Project_Team1/data/books_merged_clean.parquet"
    data = load_and_clean_dataset(file_path)
    # call model
    X, lsa_model = train_lsa_model(data, combined_column='review_combined')
    # save the model
    save_model(lsa_model, "SVD_recommendations2.pkl")

if __name__ == "__main__":
    main()