import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random
from torch.nn.utils.rnn import pad_sequence

class UserInteractionDataset(Dataset):
    def __init__(self, file_path, max_len, mask_prob=0.15, mask_token=0):
        self.data = pd.read_parquet(file_path)
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.item_id_mapping = {item: idx + 1 for idx, item in enumerate(sorted(set(self.data['Item_sequence'].explode())))}  # Mapping items to unique integers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data['Item_sequence'].iloc[idx]
        sequence = [self.item_id_mapping[item] for item in sequence]  # Map items to integers
        masked_sequence, labels = self.mask_sequence(sequence)
        # Convert lists to tensors and pad them to the maximum sequence length
        masked_sequence = torch.tensor(masked_sequence, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        # Pad sequences if they are shorter than max_len
        if len(masked_sequence) < self.max_len:
            masked_sequence = torch.cat((torch.full((self.max_len - len(masked_sequence),), self.mask_token, dtype=torch.long), masked_sequence))
            labels = torch.cat((torch.full((self.max_len - len(labels),), 0, dtype=torch.long), labels))
        return masked_sequence, labels

    def mask_sequence(self, sequence):
        masked_sequence = []
        labels = []
        for item in sequence:
            if random.random() < self.mask_prob:
                masked_sequence.append(self.mask_token)
                labels.append(item)
            else:
                masked_sequence.append(item)
                labels.append(0)  # No item to predict here
        return masked_sequence, labels



# Path to your parquet file containing the sequences
parquet_file_path = '/home/ubuntu/NLP_Project_Team1/data/user_sequences.parquet'

# Define the maximum length of sequences based on your data analysis
MAX_SEQUENCE_LENGTH = 512  # You might need to adjust this based on your actual data

interaction_dataset = UserInteractionDataset(parquet_file_path, MAX_SEQUENCE_LENGTH, mask_prob=0.2, mask_token=999999)

# Create a DataLoader to handle batching
dataloader = DataLoader(interaction_dataset, batch_size=32, shuffle=True, num_workers=4)  # Adjust num_workers based on your system's capabilities

# Check the output from the dataloader
for data, labels in dataloader:
    print("Data (masked sequence):", data)
    print("Labels (original items):", labels)
    break  # Only display the first batch for checking
