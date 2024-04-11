#%%
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
import pickle
from scipy.sparse import load_npz
#import pbd

# Function to load and clean the data
def load_and_clean_dataset(file_path, summary_column='review/summary', text_column='review/text', id_column='Id'):
    print("Loading and cleaning data...")
    # Load the dataset from a Parquet file
    data = pd.read_parquet(file_path)
    # Fill missing values for certain columns
    data.fillna({'review/title': '', 'review/summary': ''}, inplace=True)
    # Combine text data
    data['review_combined'] = data[summary_column] + ' ' + data[text_column]
    # Drop rows with missing values for the specified columns
    cleaned_data = data.dropna(subset=['review_combined'], how='any')
    return cleaned_data, data[id_column]


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
def train_lsa_model(data, ids, combined_column='review_combined', max_df=0.5, min_df=2, n_components=100, random_state=42):
    print("Training LSA model...")
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words='english', use_idf=True)
    svd_model = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=random_state)
    lsa_model = make_pipeline(vectorizer, svd_model)
    X = lsa_model.fit_transform(data[combined_column])
    # Combine the vectors with their respective IDs
    vectors_with_ids = pd.DataFrame(X, index=ids)
    # Save the vectors and their IDs
    vectors_with_ids.to_pickle('lsa_vectors_with_ids.pkl')
    return vectors_with_ids, lsa_model

def align_vectors_with_interaction_data(vectors_with_ids_path, interaction_data_path):
    vectors_with_ids = pd.read_pickle(vectors_with_ids_path)
    interaction_data = pd.read_csv(interaction_data_path)  # Load your interaction data
    # Merge based on item_id, assuming interaction data has an 'item_id' column
    aligned_data = interaction_data.merge(vectors_with_ids, left_on='Id', right_index=True, how='left')
    return aligned_data

def validate_alignment(aligned_data):
    # Check a random sample to verify alignment manually or through automated checks
    sample = aligned_data.sample(n=5)
    print(sample[['Id', 'item_description', 'vector']])  # Customize column names as per your dataset


# Function to save the trained model
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

from sklearn.metrics.pairwise import cosine_similarity

def compute_item_similarities(vectors):
    print("Computing item similarities...")
    # Compute cosine similarity matrix from LSA vectors
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix

def hybrid_recommendation(user_id, user_item_matrix, item_similarity_matrix, user_ids_map, item_ids_map, top_n=10):
    print("Making recommendations for user:", user_id)
    # Convert user_id to the internal index used in the matrix
    user_index = user_ids_map.get(user_id, None)
    if user_index is None:
        return "User ID not found."
    
    # Extract user row from sparse matrix
    user_interactions = user_item_matrix.getrow(user_index).toarray()

    # Compute scores
    scores = np.dot(item_similarity_matrix, user_interactions.T).flatten()

    # Get top-n item indices
    top_item_indices = np.argsort(scores)[::-1][:top_n]

    # Convert indices back to item IDs
    top_item_ids = [item_ids_map.inverse[idx] for idx in top_item_indices]  # Assuming `item_ids_map` is a bidirectional map
    
    return top_item_ids


#%%

print("Loading and cleaning data...")
data, item_ids = load_and_clean_dataset("/home/ubuntu/NLP_Project_Team1/data/books_merged_clean.parquet")
print("Data loaded and cleaned.")

#%%
print("Preprocessing text for all data entries...")
data['review_combined'] = data['review_combined'].apply(preprocess_text)
print("Text preprocessing complete.")
#%%
print("Training LSA model...")
vectors_with_ids, lsa_model = train_lsa_model(data, item_ids, 'review_combined')
print("LSA model trained.")
#%%
print("Loading user-item interaction matrix...")
interaction_matrix_path = "/home/ubuntu/NLP_Project_Team1/data/user_item_interaction_matrix.npz"
user_item_matrix = load_npz(interaction_matrix_path)
print("User-item interaction matrix loaded.")
#%%
print("Computing item-item similarities...")
item_similarity_matrix = compute_item_similarities(vectors_with_ids.values)
print("Item-item similarities computed.")
#%%
print("Making recommendations...")
user_id = 'example_user_id'
recommendations = hybrid_recommendation(user_id, user_item_matrix, item_similarity_matrix)
print("Recommendations for User:", user_id, recommendations)



# def main():
#     print("Starting program...")
#     try:
#         print("Loading and cleaning data...")
#         data, item_ids = load_and_clean_dataset("/home/ubuntu/NLP_Project_Team1/data/books_merged_clean.parquet")
#         print("Data loaded and cleaned.")
        
#         print("Preprocessing text for all data entries...")
#         data['review_combined'] = data['review_combined'].apply(preprocess_text)
#         print("Text preprocessing complete.")
        
#         print("Training LSA model...")
#         vectors_with_ids, lsa_model = train_lsa_model(data, item_ids, 'review_combined')
#         print("LSA model trained.")
        
#         print("Loading user-item interaction matrix...")
#         interaction_matrix_path = "/home/ubuntu/NLP_Project_Team1/data/user_item_interaction_matrix.npz"
#         user_item_matrix = load_npz(interaction_matrix_path)
#         print("User-item interaction matrix loaded.")

#         print("Computing item-item similarities...")
#         item_similarity_matrix = compute_item_similarities(vectors_with_ids.values)
#         print("Item-item similarities computed.")

#         print("Making recommendations...")
#         user_id = 'example_user_id'
#         recommendations = hybrid_recommendation(user_id, user_item_matrix, item_similarity_matrix)
#         print("Recommendations for User:", user_id, recommendations)

#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()
