import pandas as pd
import numpy as np

# Load the datasets
# books_data_path = 'data/books_data.csv'  # Update the path accordingly
# books_ratings_path = 'data/Books_rating.csv'  # Update the path accordingly

# books_data = pd.read_csv(books_data_path)
# books_ratings = pd.read_csv(books_ratings_path)

# # Basic information
# print("Books Data Info:")
# print(books_data.info())
# print("\nBooks Ratings Info:")
# print(books_ratings.info())

# # Display the first few rows to understand the data better
# print("\nBooks Data Preview:")
# print(books_data.head())
# print("\nBooks Ratings Preview:")
# print(books_ratings.head())

# # Check for missing values
# print("\nMissing Values in Books Data:")
# print(books_data.isnull().sum())
# print("\nMissing Values in Books Ratings:")
# print(books_ratings.isnull().sum())

# # Basic statistics
# print("\nBooks Data Description:")
# print(books_data.describe())
# print("\nBooks Ratings Description:")
# print(books_ratings.describe())


# # Correcting the Ratings Distribution analysis
# print("\nReview Scores Distribution:")
# print(books_ratings['review/score'].value_counts(normalize=True))

# # Investigate the most reviewed books
# most_reviewed_books = books_ratings['Title'].value_counts().head(10)
# print("\nMost Reviewed Books:")
# print(most_reviewed_books)

# # Average review score per book
# average_book_score = books_ratings.groupby('Title')['review/score'].mean().sort_values(ascending=False).head(10)
# print("\nBooks with Highest Average Score:")
# print(average_book_score)

# # Investigate the distribution of review counts per user
# review_count_per_user = books_ratings['User_id'].value_counts()
# print("\nDistribution of Review Counts per User:")
# print(review_count_per_user.describe())

# # Investigate the distribution of books across categories (if applicable)
# if 'categories' in books_data.columns:
#     books_per_category = books_data['categories'].value_counts().head(10)
#     print("\nDistribution of Books per Category:")
#     print(books_per_category)

# # Further analyses could involve text processing on the review texts to understand sentiment,
# # clustering books based on their descriptions or categories to identify similar books, etc.


# #books_merged = pd.merge(books_data, books_ratings, on='Title')

# # Optionally, save the merged dataset to a new CSV file
# #books_merged.to_csv('data/books_merged.csv', index=False)

# # Read the CSV file into a DataFrame
# books_merged = pd.read_csv('data/books_merged.csv')

# # Display the first few rows of the DataFrame
# print(books_merged.head())
# # Display unique user IDs present in the dataset
# unique_user_ids = books_merged['User_id'].unique()
# print("Unique User IDs:")
# print(unique_user_ids)

# # Now, you can choose a real user ID from the list and filter the DataFrame accordingly
# # For example, let's say we choose the first user ID from the list
# chosen_user_id = unique_user_ids[0]

# # Filter the DataFrame to find reviews made by the chosen user
# user_reviews = books_merged[books_merged['User_id'] == chosen_user_id]

# # Display the first few rows of the user's reviews
# print("\nReviews made by user with ID:", chosen_user_id)
# print(user_reviews.head())


import pandas as pd

data = pd.read_parquet('/home/ubuntu/NLP_Project_Team1/data/books_merged_clean.parquet')
print(data.columns)

# Set display options
pd.set_option('display.max_columns', None)  # Ensures all columns are shown
pd.set_option('display.max_colwidth', None)  # Removes truncation of column content
pd.set_option('display.width', None)  # Uses maximum width to display each row


# Print the last five rows of the DataFrame
print(data.tail(5))