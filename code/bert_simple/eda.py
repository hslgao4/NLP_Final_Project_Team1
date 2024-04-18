import pandas as pd

# Load your data
data = pd.read_parquet('/home/ubuntu/NLP_Project_Team1/data/user_sequences.parquet')

# Suppose your column with text/tokens is named 'Item_sequence'
# Explode the sequence into individual items if they are in a list format
item_sequences = data['Item_sequence'].explode()

# Get unique items
unique_items = item_sequences.unique()

# Calculate vocab size; add 1 for potential unknown or special tokens
vocab_size = len(unique_items) + 1

print("Vocabulary Size:", vocab_size)
