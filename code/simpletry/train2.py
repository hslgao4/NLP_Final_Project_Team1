import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score


def load_and_clean_dataset(file_path, summary_column='review/summary', text_column='review/text', min_reviews=10, min_score=4):
    print("Loading data...")
    data = pd.read_parquet(file_path)

    print("Cleaning data...")
    if 'review_combined' not in data.columns:
        data.fillna({summary_column: '', text_column: ''}, inplace=True)
        data['review_combined'] = data[summary_column] + ' ' + data[text_column]

    # Filter out entries based on review score
    data = data[data['review/score'] >= min_score]

    # Group by user and filter out those with fewer than min_reviews
    user_review_counts = data['User_id'].value_counts()
    users_with_enough_reviews = user_review_counts[user_review_counts >= min_reviews].index
    filtered_data = data[data['User_id'].isin(users_with_enough_reviews)]
    
    print("Data filtered. Only positive reviews retained. Users with at least", min_reviews, "reviews retained.")
    print("Columns available:", filtered_data.columns)
    return filtered_data



def get_bert_embeddings(data, tokenizer, model, text_column='review_combined', batch_size=500, device='cuda', save_path='embeddings'):
    print("Starting BERT embeddings generation...")
    all_embeddings = []

    # Ensure directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    total_batches = len(data) // batch_size + (len(data) % batch_size != 0)
    print(f"Total number of batches: {total_batches}")

    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        encoded_input = tokenizer(batch[text_column].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=512)
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}  # Ensure input tensors are on GPU
        print(f"Processing batch {i // batch_size + 1}/{total_batches}")
        with torch.no_grad():
            output = model(**encoded_input)
        embeddings = output.last_hidden_state[:, 0, :].detach().cpu().numpy()  # Move to CPU after processing if necessary

        # Save embeddings of the current batch
        np.save(os.path.join(save_path, f'batch_{i // batch_size + 1}.npy'), embeddings)
        all_embeddings.append(embeddings)

    # Optionally, combine and save all embeddings into one file at the end
    all_embeddings = np.vstack(all_embeddings)
    np.save(os.path.join(save_path, 'all_embeddings.npy'), all_embeddings)
    print("BERT embeddings generated and saved successfully.")
    return all_embeddings



def find_similar_books(input_review, book_embeddings, books_df, tokenizer, model, top_n=10, device='cuda'):
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


from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

def main():
    print("Checking GPU availability...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Initializing the tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    print("Tokenizer and model loaded.")

    # Load and clean dataset
    data = load_and_clean_dataset("data/books_merged_clean.parquet")

    # Check for existing training and testing data
    train_embeddings_file = 'train_embeddings.npy'
    test_embeddings_file = 'test_embeddings.npy'
    train_data_file = 'train_books.csv'
    test_data_file = 'test_books.csv'

    if os.path.exists(train_embeddings_file) and os.path.exists(test_embeddings_file) \
       and os.path.exists(train_data_file) and os.path.exists(test_data_file):
        print("Loading existing training and testing data...")
        train_embeddings = np.load(train_embeddings_file)
        test_embeddings = np.load(test_embeddings_file)
        train_data = pd.read_csv(train_data_file)
        test_data = pd.read_csv(test_data_file)
    else:
        print("Splitting data and generating embeddings...")
        train_data, test_data, train_embeddings, test_embeddings = train_test_split(
            data, get_or_load_embeddings(data, tokenizer, model, device), test_size=0.2, random_state=42
        )
        np.save(train_embeddings_file, train_embeddings)
        np.save(test_embeddings_file, test_embeddings)
        train_data.to_csv(train_data_file, index=False)
        test_data.to_csv(test_data_file, index=False)

    # Example usage
    input_review = "I love this book about AI and technology, very insightful."
    print("Processing input review for recommendations...")
    recommended_books_df = find_similar_books(input_review, test_embeddings, test_data, tokenizer, model, device=device)
    recommended_book_ids = recommended_books_df['Id'].tolist()
    actual_liked_books = test_data[test_data['review/score'] >= 4]['Id'].tolist()

    # Evaluate performance
    precision, recall, f1 = evaluate_performance(recommended_book_ids, actual_liked_books)
    print("Recommended books:")
    print(recommended_books_df)
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

if __name__ == "__main__":
    main()