import pandas as pd
import cudf
import nvtabular as nvt
from nvtabular.ops import Categorify, Normalize, LogOp, LambdaOp, Rename, AddMetadata
from merlin.schema import Tags

# Assuming your environment is already set up for CUDA operations with cuDF and NVTabular

# Step 1: Reading the Parquet file and converting to cuDF DataFrame
# If the dataset is large, consider loading it directly into cuDF if it fits into memory
file_path = '/home/ubuntu/.cache/NLP_Project_Team1/data/books_merged_clean.parquet'
gdf_books = cudf.read_parquet(file_path)

# Step 2: Defining the NVTabular workflow for preprocessing
cat_features = ['Title', 'authors', 'publisher', 'categories'] >> Categorify()

# Assuming 'review/time' is a Unix timestamp; converting to datetime
# Note: This operation is directly applied on the cuDF DataFrame
gdf_books['review_datetime'] = cudf.to_datetime(gdf_books['review/time'], unit='s')
review_dayofweek = ['review_datetime'] >> LambdaOp(lambda col: col.dt.weekday) >> AddMetadata(tags=[Tags.CATEGORICAL])

cont_features = ['ratingsCount'] >> LogOp() >> Normalize() >> Rename(postfix='_norm')

# Combine features for the workflow
features = cat_features + review_dayofweek + cont_features

# Initialize and apply the NVTabular workflow
# NVTabular handles large datasets by processing them in chunks, effectively batching
workflow = nvt.Workflow(features)
dataset = nvt.Dataset(gdf_books, engine='parquet', part_mem_fraction=0.1) # Adjust part_mem_fraction as needed

# Note: Consider specifying the output path relevant to your environment
output_path = '/home/ubuntu/.cache/NLP_Project_Team1/data/processed_data'
workflow.fit_transform(dataset).to_parquet(output_path=output_path)
