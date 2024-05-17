import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import re
from Bert_train import *

# Check GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get category from Train.py
model_path = "../Code/result/checkpoint-4656"
text = 'Navigating the tumultuous waters of adolescence has never been more captivating than in this poignant young adult novel that explores the trials and triumphs of growing up. From first love to friendship struggles, the story delves into the complex emotions and experiences of teenage life. With its relatable characters and authentic voice, it resonates with readers of all ages, capturing the essence of youth with honesty and empathy. A coming-of-age tale that speaks to the heart and soul of every teenager.'

category = model_predict(model_path, text)


# Import cleaned data
path = '../Code/data/final_df.parquet'
dff = pd.read_parquet(path)

df = dff.copy()
df = df[df['categories'] == category].reset_index(drop=True)

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Generate embeddings - feature extraction
def generate_embedding(text):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    tokens = {key: val.to(device) for key, val in tokens.items()}
    with torch.no_grad():
        outputs = bert_model(**tokens)

    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().reshape(-1)

df['review_embed'] = df['review_text'].apply(generate_embedding)

#%%
def rem_deplicate(text):
    clean_text = re.sub(r"\[.*?\]", "", text)
    return re.sub(r"\(.*?\)", "", clean_text)

# Train the k-NN model
knn_model = NearestNeighbors(n_neighbors=50)
knn_model.fit(np.stack(df['review_embed'].values))

def recommend_books(new_text):
    new_embedding = generate_embedding(new_text)
    distances, indices = knn_model.kneighbors([new_embedding])
    recommended_books = df.iloc[indices[0]]
    recommended_books = recommended_books.drop_duplicates(subset=['review_summary'], keep='first')
    books = recommended_books[['Title', 'authors', 'review_score']]
    books = books.copy()
    books.Title = books.Title.apply(rem_deplicate)
    book_unique = books.drop_duplicates(subset=['Title']).copy()
    book_unique = book_unique.sort_values(by='review_score', ascending=False)
    temp = book_unique.copy()
    top6_book = temp.head(6)
    return top6_book

# Example usage
books = recommend_books(text)
print(books.to_string())