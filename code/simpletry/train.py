import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel
import torch
import pickle

# Make sure to download the required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Function to load and clean the data
def load_and_clean_dataset(file_path, summary_column='review/summary', text_column='review/text'):
    print("Loading data...")
    data = pd.read_parquet(file_path)
    print("Cleaning data...")
    data.fillna({'review/title': '', 'review/summary': ''}, inplace=True)
    data['review_combined'] = data[summary_column] + ' ' + data[text_column]
    cleaned_data = data.dropna(subset=['review_combined'], how='any')
    print("Data loaded and cleaned.")
    return cleaned_data


# Function to generate BERT embeddings
def generate_bert_embeddings(data, combined_column='review_combined'):
    print("Loading BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    print("Generating embeddings...")
    encoded_input = tokenizer(data[combined_column].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        output = model(**encoded_input)
    embeddings = output.last_hidden_state[:, 0, :]
    print("Embeddings generated.")
    return embeddings

# Save model function remains the same

def main():
    print("Script started...")
    file_path = "data/books_merged_clean.parquet"
    
    print("Running load_and_clean_dataset function...")
    data = load_and_clean_dataset(file_path)
    
    print("Running generate_bert_embeddings function...")
    embeddings = generate_bert_embeddings(data, combined_column='review_combined')
    
    print("Saving embeddings...")
    with open('bert_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings.cpu().numpy(), f)
    print("Embeddings saved.")

    print("Script finished.")

if __name__ == "__main__":
    main()
