import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import logging

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_clean_dataset(file_path):
    try:
        logging.info("Loading data...")
        data = pd.read_parquet(file_path)
        logging.info("Cleaning data...")
        data['review_combined'] = data['description'].fillna('') + ' ' + data['review_text'].fillna('')
        logging.info("Data loaded and combined.")
        return data
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

from transformers import BertModel


def get_bert_embeddings(data, tokenizer, model, text_column='review_combined', batch_size=500, device='cuda', save_path='embeddings'):
    print("Starting BERT embeddings generation...")
    all_embeddings = []

    # Ensure directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    total_batches = len(data) // batch_size + (len(data) % batch_size != 0)
    print(f"Total number of batches: {total_batches}")

    progress_bar = tqdm(total=total_batches, desc="Generating Embeddings")  # Set up the tqdm progress bar

    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        encoded_input = tokenizer(batch[text_column].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=512)
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            # Make sure to use BertModel here
            output = model(**encoded_input)
        embeddings = output.last_hidden_state[:, 0, :].detach().cpu().numpy()  # Correct attribute for embeddings

        # Save embeddings of the current batch
        np.save(os.path.join(save_path, f'batch_{i // batch_size + 1}.npy'), embeddings)
        all_embeddings.append(embeddings)
        progress_bar.update(1)  # Update the progress bar after each batch

    progress_bar.close()  # Close the progress bar after all batches are processed

    # Combine and save all embeddings into one file
    all_embeddings = np.vstack(all_embeddings)
    np.save(os.path.join(save_path, 'all_embeddings.npy'), all_embeddings)
    print("BERT embeddings generated and saved successfully.")
    return all_embeddings


def find_similar_books(input_review, book_embeddings, books_df, tokenizer, model, top_n=20, device='cuda'):
    print("Finding similar books...")
    # Tokenize input review and ensure tensor is on the correct device
    input_encoded = tokenizer(input_review, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_encoded = {key: val.to(device) for key, val in input_encoded.items()}  # Move to GPU
    print("Input review encoded.")

    with torch.no_grad():
        input_embedding = model(**input_encoded).last_hidden_state[:, 0, :]
    print("Input review embedding generated.")

    # Ensure book_embeddings are also moved to the same device temporarily for similarity calculation
    book_embeddings_tensor = torch.tensor(book_embeddings).to(device)
    similarities = cosine_similarity(input_embedding.cpu().numpy(), book_embeddings_tensor.cpu().numpy())  # Move back to CPU for sklearn operations
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    
    print(f"Most similar books indices: {top_indices}")
    print(f"Total books available: {len(books_df)}")

    # Ensure the indices do not exceed the DataFrame length
    top_indices = top_indices[top_indices < len(books_df)]

    if len(top_indices) == 0:
        print("No similar books found within the provided index range.")
        return pd.DataFrame()  # Return an empty DataFrame if no indices are valid

    print("Most similar books found.")
    return books_df.iloc[top_indices]


def get_or_load_embeddings(data, tokenizer, model, device='cuda', embeddings_path='all_embeddings.npy'):
    if os.path.exists(embeddings_path):
        print("Loading existing embeddings from file.")
        return np.load(embeddings_path)
    else:
        print("Embeddings file not found, generating embeddings.")
        return get_bert_embeddings(data, tokenizer, model, device=device)

def evaluate_performance(recommended_books, actual_liked_books):
    true_positives = set(recommended_books) & set(actual_liked_books)
    false_negatives = set(actual_liked_books) - set(recommended_books)
    precision = len(true_positives) / len(recommended_books) if recommended_books else 0
    recall = len(true_positives) / len(actual_liked_books) if actual_liked_books else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    print(f"True Positives (Matched Recommendations): {len(true_positives)}")
    print(f"False Negatives (Missed Relevant Books): {len(false_negatives)}")
    return precision, recall, f1


from tqdm import tqdm  # Import the tqdm function

from transformers import BertForSequenceClassification

def fine_tune_bert(data, tokenizer, device):
    print("Fine-tuning BERT...")
    texts = data['review_combined'].tolist()
    labels = (data['review/score'] > 4).astype(int).tolist()

    # Prepare the data for training
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_attention_mask=True)
    dataset = TensorDataset(torch.tensor(encodings['input_ids']), torch.tensor(encodings['attention_mask']), torch.tensor(labels))

    # Split data into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Initialize the model for sequence classification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

    # Setup optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 3)

    # Training loop
    for epoch in range(3):
        model.train()
        train_progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
        for batch in train_progress_bar:
            inputs, attention_mask, labels = [x.to(device) for x in batch]
            model.zero_grad()
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_progress_bar.set_postfix({'Training Loss': f"{loss.item():.4f}"})

        # Validation phase
        model.eval()
        val_loss = 0
        val_progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False)
        for batch in val_progress_bar:
            inputs, attention_mask, labels = [x.to(device) for x in batch]
            with torch.no_grad():
                outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
            val_progress_bar.set_postfix({'Validation Loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} completed. Avg Validation Loss: {avg_val_loss:.4f}")

    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load the cleaned dataset
    data = load_and_clean_dataset("data/final_df.parquet")
    model_path = 'bert_finetuned_model.pth'

    # File paths
    train_embeddings_file = 'train_embeddings.npy'
    test_embeddings_file = 'test_embeddings.npy'
    train_data_file = 'train_books.csv'
    test_data_file = 'test_books.csv'

    # Initialize the model for embeddings regardless of the condition
    model_for_embeddings = BertModel.from_pretrained('bert-base-uncased').to(device)

    # Check if the fine-tuned model exists and load it
    if os.path.exists(model_path):
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        print("Loading existing training and testing data...")
        train_embeddings = np.load(train_embeddings_file)
        test_embeddings = np.load(test_embeddings_file)
        train_data = pd.read_csv(train_data_file)
        test_data = pd.read_csv(test_data_file)
    else:
        print("Generating new splits and embeddings...")
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        # Generate embeddings
        embeddings = get_bert_embeddings(data, tokenizer, model_for_embeddings, device=device)

        # Save new data splits and embeddings
        np.save(train_embeddings_file, embeddings)
        np.save(test_embeddings_file, embeddings)
        train_data.to_csv(train_data_file, index=False)
        test_data.to_csv(test_data_file, index=False)

    # Example usage for recommendation and evaluation
    input_review = "I love this book about AI and technology."
    recommended_books_df = find_similar_books(input_review, test_embeddings, test_data, tokenizer, model_for_embeddings, device=device)
    recommended_book_ids = recommended_books_df['Id'].tolist()
    actual_liked_books = test_data[test_data['review/score'] >= 4]['Id'].tolist()
    precision, recall, f1 = evaluate_performance(recommended_book_ids, actual_liked_books)
    print("Recommended books:", recommended_books_df)
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

if __name__ == "__main__":
    main()
