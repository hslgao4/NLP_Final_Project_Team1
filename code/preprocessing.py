
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


# clean all text
def clean_text(text):
    # Lowercase text
    # text = text.lower()
    # Remove parentheses and words within them
    text = re.sub(r'\([^)]*\)', '', text)
    # Remove punctuation (except apostrophes)
    text = re.sub(r'[^a-zA-Z0-9\'.]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Replace "nan" with "N/A"
    text = text.replace("nan", "N/A")
    
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



