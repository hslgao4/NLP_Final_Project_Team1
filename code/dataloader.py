import pandas as pd
import requests
from io import BytesIO

import os
import pandas as pd
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

import os
import pandas as pd
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()

# Specify the dataset name
dataset_name = 'mohamedbakhet/amazon-books-reviews'

# Define the path for the 'data' folder
data_path = './data'

# Create 'data' directory if it doesn't exist
if not os.path.exists(data_path):
    os.makedirs(data_path)

# Download dataset (this will download a zip file containing the dataset into the 'data' folder)
api.dataset_download_files(dataset_name, path=data_path, unzip=False)

# Path to the downloaded zip file
zip_file_path = os.path.join(data_path, 'amazon-books-reviews.zip')

# Unzipping the dataset inside the 'data' directory
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(data_path)

# Assuming the dataset contains 'books_data.csv' and 'Books_rating.csv'
# Construct paths to the dataset files
books_data_path = os.path.join(data_path, 'books_data.csv')
books_ratings_path = os.path.join(data_path, 'Books_rating.csv')

# Load the datasets into Pandas DataFrames
books_data = pd.read_csv(books_data_path)
books_ratings = pd.read_csv(books_ratings_path)

# Merge the data on the 'Title' column
books_merged = pd.merge(books_data, books_ratings, on='Title', how='inner')

# Save the merged dataset to a new CSV file inside the 'data' folder
merged_data_path = os.path.join(data_path, 'books_merged.csv')
books_merged.to_csv(merged_data_path, index=False)



# Function to perform basic EDA
def basic_eda(df):
    print("Info:")
    print(df.info())
    print("\nHead:")
    print(df.head())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDescription:")
    print(df.describe())

# Perform basic EDA
print("\nBooks Data EDA")
basic_eda(books_data)
print("\nBooks Ratings EDA")
<<<<<<< HEAD
basic_eda(books_ratings)

print("\nBooks Ratings EDA")
basic_eda(books_merged)




=======
basic_eda(books_ratings)
>>>>>>> b05b3222f96ff5431fc692bbd02b61a94303842e
