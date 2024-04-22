import pandas as pd
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm import tqdm

def load_and_clean_dataset(file_path):
    print("Loading data...")
    data = pd.read_parquet(file_path)
    print("Cleaning data...")
    data['review_combined'] = data['description'].fillna('') + ' ' + data['review_text'].fillna('')
    print("Data loaded and combined.")
    return data


def get_gpt_embeddings(data, tokenizer, model, text_column='review_combined', batch_size=16, device='cuda', save_path='gpt_embeddings'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)  # Ensure the directory exists

    print("Starting GPT embeddings generation...")
    all_embeddings = []
    total_batches = len(data) // batch_size + (len(data) % batch_size != 0)
    print(f"Total number of batches: {total_batches}")

    progress_bar = tqdm(total=total_batches, desc="Generating Embeddings")
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        encoded_input = tokenizer(batch[text_column].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=512)
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            output = model(**encoded_input)
        
        embeddings = output.last_hidden_state[:, 0, :].detach().cpu().numpy()
        np.save(os.path.join(save_path, f'batch_{i // batch_size + 1}.npy'), embeddings)
        all_embeddings.append(embeddings)
        progress_bar.update(1)
    
    progress_bar.close()
    all_embeddings = np.vstack(all_embeddings)
    np.save(os.path.join(save_path, 'all_embeddings.npy'), all_embeddings)
    print("GPT embeddings generated and saved successfully.")
    return all_embeddings

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2').to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_and_clean_dataset("data/final_df.parquet")
    embeddings = get_gpt_embeddings(data, tokenizer, model, device=device)

if __name__ == "__main__":
    main()
