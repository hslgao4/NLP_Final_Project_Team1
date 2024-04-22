import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class GPT2ForSequenceClassification(torch.nn.Module):
    """Custom GPT-2 model for sequence classification."""
    def __init__(self, num_labels=2):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.gpt2.config.n_embd, num_labels)
        print("GPT-2 model initialized for sequence classification.")

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        output = outputs.last_hidden_state[:, 0]  # Take the first token's embedding
        output = self.dropout(output)
        logits = self.classifier(output)
        return logits

def find_similar_books(input_review, embeddings, df, tokenizer, model, device):
    """Find books similar to the input review."""
    print("Finding similar books based on the input review...")
    inputs = tokenizer(input_review, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask)
    
    distances = torch.nn.functional.cosine_similarity(output, embeddings, dim=1)
    top_indices = distances.argsort(descending=True)[:10]
    return df.iloc[top_indices.cpu().numpy()]

def evaluate_performance(recommended_ids, actual_liked_ids):
    """Calculate precision, recall, and F1-score."""
    print("Evaluating performance...")
    true_positives = set(recommended_ids) & set(actual_liked_ids)
    false_positives = set(recommended_ids) - set(actual_liked_ids)
    false_negatives = set(actual_liked_ids) - set(recommended_ids)
    precision = len(true_positives) / len(recommended_ids) if recommended_ids else 0
    recall = len(true_positives) / len(actual_liked_ids) if actual_liked_ids else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

def load_and_prepare_data(file_path):
    """Load and clean dataset, then split into training and testing datasets."""
    print("Loading data...")
    data = pd.read_parquet(file_path)
    print("Cleaning data...")
    data['review_combined'] = data['description'].fillna('') + ' ' + data['review_text'].fillna('')
    print("Data loaded and combined.")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2ForSequenceClassification(num_labels=2).to(device)

    # Load and prepare data
    train_data, test_data = load_and_prepare_data("data/final_df.parquet")
    embeddings = np.load("gpt_embeddings/all_embeddings.npy")
    embeddings_tensor = torch.tensor(embeddings).to(device)

    # Recommendation and evaluation
    input_review = "I love this book about AI and technology, very insightful."
    recommended_books_df = find_similar_books(input_review, embeddings_tensor, test_data, tokenizer, model, device)
    recommended_book_ids = recommended_books_df['Id'].tolist()
    actual_liked_books = test_data[test_data['review/score'] >= 4]['Id'].tolist()
    
    precision, recall, f1 = evaluate_performance(recommended_book_ids, actual_liked_books)
    print("Recommended books:", recommended_books_df)
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

if __name__ == "__main__":
    main()
