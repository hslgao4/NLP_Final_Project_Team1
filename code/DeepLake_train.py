import os
import numpy as np
import pandas as pd
from deeplake import VectorStore
from sentence_transformers import SentenceTransformer

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_embeddings(search_results, vector_store):
    print("Visualizing similar books...")
    print("Type of search_results:", type(search_results))
    print("First element if it's a list:", search_results[0] if isinstance(search_results, list) else "Not a list")

    
    # Check if search_results is a list of dictionaries
    if isinstance(search_results, list) and all(isinstance(res, dict) for res in search_results):
        ids = [result['id'] for result in search_results]
        # Retrieve embeddings using the extracted identifiers
        embeddings = vector_store.get(ids)
        
        # Use t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings)

        # Plot the embeddings in a 2D space
        plt.figure(figsize=(12, 8))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue')

        # Annotate the points with their respective titles
        for i, title in enumerate([result['Title'] for result in search_results]):
            plt.annotate(title, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

        plt.title("t-SNE Visualization of Book Embeddings")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.show()
    else:
        print("The structure of search_results is not as expected.")





def load_and_clean_dataset(file_path):
    print("Loading data...")
    data = pd.read_parquet(file_path)
    print("Data loaded.")

    # Standardize titles and authors by capitalizing the first letter of each word
    data['Title'] = data['Title'].str.title().str.strip()
    data['authors'] = data['authors'].str.title().str.strip()

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

    print(f"Total data size: {len(data)}")
    print(f"Total number of batches: {total_batches}")

    seen_titles = set()  # To avoid processing the same title multiple times if it appears in multiple batches
    for batch_index in range(0, len(data), batch_size):
        batch = data.iloc[batch_index:batch_index + batch_size]
        batch = batch[~batch['Title'].isin(seen_titles)]
        seen_titles.update(batch['Title'].tolist())

        print(f"Processing batch {batch_index // batch_size + 1}/{total_batches} with {len(batch)} items.")

        texts = batch['description'].tolist()  # Using book descriptions for embedding
        embeddings = model.encode(texts, show_progress_bar=False)

        metadata = batch[['Title', 'authors', 'review/score']].to_dict(orient='records')
        vector_store.add(embedding=embeddings, metadata=metadata, text=texts)

    vector_store.commit()
    print("VectorStore created and data added.")
    return vector_store


def find_similar_books(input_text, vector_store, model, top_n=10):
    print("Finding similar books...")
    input_embedding = model.encode([input_text])[0]
    search_results = vector_store.search(embedding=input_embedding, k=top_n, distance_metric='COS')
    if search_results:
        similar_books = pd.DataFrame(search_results['metadata'])
        # Capitalize the first letter of each word for title and authors
        similar_books['Title'] = similar_books['Title'].str.title()
        similar_books['authors'] = similar_books['authors'].str.title()
        print("Most similar books found:")
        print(similar_books[['Title', 'authors', 'review/score']].head(6))
        visualize_embeddings(search_results, vector_store)

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
    if not recommended_books_df.empty:
        visualize_embeddings(recommended_books_df.to_dict(orient='list'), vector_store)
    
if __name__ == "__main__":
    main()