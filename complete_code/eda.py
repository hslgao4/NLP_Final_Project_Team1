import pandas as pd
import numpy as np
import requests
from io import BytesIO, StringIO

# Function to download data from a URL
def download_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print("Failed to download data")
        return None

# Load the datasets from URL
books_data_url = 'https://raw.githubusercontent.com/cp-bailey/NLP_Project_Team1/main/data/books_data.csv'
books_ratings_url = 'https://raw.githubusercontent.com/cp-bailey/NLP_Project_Team1/main/data/Books_rating.csv'

books_data_content = download_data(books_data_url)
books_ratings_content = download_data(books_ratings_url)

if books_data_content and books_ratings_content:
    # Convert bytes to pandas DataFrame
    books_data = pd.read_csv(BytesIO(books_data_content))
    books_ratings = pd.read_csv(BytesIO(books_ratings_content))

# Basic information
print("Books Data Info:")
print(books_data.info())
print("\nBooks Ratings Info:")
print(books_ratings.info())

# Display the first few rows to understand the data better
print("\nBooks Data Preview:")
print(books_data.head())
print("\nBooks Ratings Preview:")
print(books_ratings.head())

# Check for missing values
print("\nMissing Values in Books Data:")
print(books_data.isnull().sum())
print("\nMissing Values in Books Ratings:")
print(books_ratings.isnull().sum())

# Basic statistics
print("\nBooks Data Description:")
print(books_data.describe())
print("\nBooks Ratings Description:")
print(books_ratings.describe())


# Correcting the Ratings Distribution analysis
print("\nReview Scores Distribution:")
print(books_ratings['review/score'].value_counts(normalize=True))

# Investigate the most reviewed books
most_reviewed_books = books_ratings['Title'].value_counts().head(10)
print("\nMost Reviewed Books:")
print(most_reviewed_books)

# Average review score per book
average_book_score = books_ratings.groupby('Title')['review/score'].mean().sort_values(ascending=False).head(10)
print("\nBooks with Highest Average Score:")
print(average_book_score)

# Investigate the distribution of review counts per user
review_count_per_user = books_ratings['User_id'].value_counts()
print("\nDistribution of Review Counts per User:")
print(review_count_per_user.describe())

# Investigate the distribution of books across categories (if applicable)
if 'categories' in books_data.columns:
    books_per_category = books_data['categories'].value_counts().head(10)
    print("\nDistribution of Books per Category:")
    print(books_per_category)

# Further analyses could involve text processing on the review texts to understand sentiment,
# clustering books based on their descriptions or categories to identify similar books, etc.


books_merged = pd.merge(books_data, books_ratings, on='Title')

# Optionally, save the merged dataset to a new CSV file
books_merged.to_csv('data/books_merged.csv', index=False)

# Read the CSV file into a DataFrame
books_merged = pd.read_csv('data/books_merged.csv')

# Display the first few rows of the DataFrame
print(books_merged.head())
# Display unique user IDs present in the dataset
unique_user_ids = books_merged['User_id'].unique()
print("Unique User IDs:")
print(unique_user_ids)

# Now, you can choose a real user ID from the list and filter the DataFrame accordingly
# For example, let's say we choose the first user ID from the list
chosen_user_id = unique_user_ids[0]

# Filter the DataFrame to find reviews made by the chosen user
user_reviews = books_merged[books_merged['User_id'] == chosen_user_id]

# Display the first few rows of the user's reviews
print("\nReviews made by user with ID:", chosen_user_id)
print(user_reviews.head())