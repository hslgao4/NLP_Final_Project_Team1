import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import random

class UserInteractionDataset(Dataset):
    def __init__(self, file_path, max_len, mask_prob=0.15, mask_token=0):
        self.data = pd.read_parquet(file_path)
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        unique_items = sorted(set(self.data['Item_sequence'].explode()))
        self.item_id_mapping = {item: idx + 1 for idx, item in enumerate(unique_items)}
        self.vocab_size = len(unique_items) + 1  # +1 for zero index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data['Item_sequence'].iloc[idx]
        sequence = [self.item_id_mapping[item] for item in sequence]
        masked_sequence, labels = self.mask_sequence(sequence)
        padded_masked_sequence = torch.full((self.max_len,), self.mask_token, dtype=torch.long)
        padded_labels = torch.full((self.max_len,), 0, dtype=torch.long)
        seq_length = min(self.max_len, len(masked_sequence))
        padded_masked_sequence[:seq_length] = torch.tensor(masked_sequence[:seq_length], dtype=torch.long)
        padded_labels[:seq_length] = torch.tensor(labels[:seq_length], dtype=torch.long)
        return padded_masked_sequence, padded_labels

    def mask_sequence(self, sequence):
        masked_sequence = []
        labels = []
        for item in sequence:
            if random.random() < self.mask_prob:
                masked_sequence.append(self.mask_token)
                labels.append(item)
            else:
                masked_sequence.append(item)
                labels.append(0)
        return masked_sequence, labels

class BERTModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_heads, n_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads),
            num_layers=n_layers
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_blocks(x)
        x = x.permute(1, 0, 2)
        x = self.classifier(x)
        return x.transpose(1, 2)

def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for seqs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}: Loss = {total_loss / len(train_loader)}')

def get_data_loaders(dataset, batch_size=16):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader

# Configuration
dataset = UserInteractionDataset('/home/ubuntu/NLP_Project_Team1/data/user_sequences.parquet', max_len=512)
train_loader, test_loader = get_data_loaders(dataset, batch_size=16)
model = BERTModel(vocab_size=dataset.vocab_size, hidden_dim=256, n_heads=8, n_layers=2, num_classes=dataset.vocab_size)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training
train(model, train_loader, criterion, optimizer, epochs=10)

# Add evaluation function if needed
