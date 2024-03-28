<<<<<<< HEAD
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
books_merged = pd.read_csv('data/books_merged.csv')

# Quick look at the data
print(books_merged.head())
print(books_merged.info())

# Check for unique values and potential inconsistencies in categorical columns
print(books_merged['Title'].value_counts())
print(books_merged['authors'].value_counts())

# Check missing values
print(books_merged.isnull().sum())

# For critical text fields, you might fill missing values with a placeholder
books_merged['Title'].fillna('Unknown', inplace=True)

# For numerical fields, you might decide to fill with the median
books_merged['ratingsCount'].fillna(books_merged['ratingsCount'].median(), inplace=True)


# Convert publishedDate to datetime
books_merged['publishedDate'] = pd.to_datetime(books_merged['publishedDate'], errors='coerce')


def clean_text(text):
    # Lowercase text
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# Correctly applying text cleaning outside the function definition
text_columns = ['Title', 'description', 'authors', 'publisher', 'categories', 'review/text', 'review/summary']
for col in text_columns:
    books_merged[col] = books_merged[col].apply(lambda x: clean_text(str(x)))

# Continue with the normalization and saving steps
scaler = MinMaxScaler()
books_merged['ratingsCount_normalized'] = scaler.fit_transform(books_merged[['ratingsCount']].fillna(0))

# Print cleaned data and check for missing values
print(books_merged.head())
print(books_merged.info())
print(books_merged.isnull().sum())

# Save the cleaned dataset as a Parquet file
books_merged.to_parquet('data/books_merged_clean.parquet')



=======
import argparse
import pandas as pd
import os

# Set up argument parsing
parser = argparse.ArgumentParser(description='Process some data.')
parser.add_argument('--data_dir', type=str, help='Input data directory', default='/data')

args = parser.parse_args()

# Use the specified data directory or default to a specific path
INPUT_DATA_DIR = args.data_dir if args.data_dir else 'data'
books_merged = pd.read_csv(os.path.join(INPUT_DATA_DIR, 'books_merged.csv'))
>>>>>>> b05b3222f96ff5431fc692bbd02b61a94303842e
