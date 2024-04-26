import os
import numpy as np
import pandas as pd
from deeplake import VectorStore
from sentence_transformers import SentenceTransformer

import pandas as pd

def load_and_clean_dataset(file_path):
    print("Loading data...")
    data = pd.read_parquet(file_path)
    print("Data loaded.")

    # Standardize titles and authors
    data['Title'] = data['Title'].str.lower().str.strip()
    data['authors'] = data['authors'].str.lower().str.strip()

    # Group by book identifiers (excluding 'Id') and select the entry with the highest review score
    data = data.groupby(['Title', 'authors']).agg({
        'review/score': 'max',  # Gets the maximum review score for each group
        'description': 'first'  # Keeps the first description found for each group
    }).reset_index()

    print("Data cleaned and filtered to include only the highest reviewed entries.")
    return data

def create_vector_store(data, model, vector_store_path="my_vector_store", batch_size=500):
    print("Creating VectorStore...")
    vector_store = VectorStore(path=vector_store_path, overwrite=True)
    total_batches = len(data) // batch_size + (len(data) % batch_size != 0)

    seen_titles = set()
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        batch = batch[~batch['Title'].isin(seen_titles)]
        seen_titles.update(batch['Title'].tolist())

        texts = batch['description'].tolist()  # Using book descriptions for embedding
        embeddings = model.encode(texts, show_progress_bar=False)
        metadata = batch[['Title', 'authors', 'review/score']].to_dict(orient='records')
        vector_store.add(embedding=embeddings, metadata=metadata, text=texts)
        print(f"Processed batch {i // batch_size + 1}/{total_batches}")
    vector_store.commit()
    print("VectorStore created and data added.")
    return vector_store

def find_similar_books(input_text, vector_store, model, top_n=10):
    print("Finding similar books...")
    input_embedding = model.encode([input_text])[0]
    search_results = vector_store.search(embedding=input_embedding, k=top_n, distance_metric='COS')
    if search_results:
        similar_books = pd.DataFrame(search_results['metadata'])
        print("Most similar books found:", similar_books[['Title', 'authors', 'review/score']])
    else:
        print("No similar books found.")
        similar_books = pd.DataFrame()
    return similar_books

def main():
    data = load_and_clean_dataset("data/final_df.parquet")
    model = SentenceTransformer('all-mpnet-base-v2')
    vector_store_path = "./my_vector_store"

    # Check if the VectorStore directory exists
    if os.path.exists(vector_store_path) and os.listdir(vector_store_path):
        print("VectorStore directory detected. Assuming it contains the necessary data and skipping recreation.")
        vector_store = VectorStore(path=vector_store_path)  # Load the existing VectorStore
    else:
        print("VectorStore directory not found or empty, creating new VectorStore...")
        vector_store = create_vector_store(data, model, vector_store_path)

    input_text = "Looking for a book about fantasy and romance"
    recommended_books_df = find_similar_books(input_text, vector_store, model)
    print("Recommended books:")
    

if __name__ == "__main__":
    main()
