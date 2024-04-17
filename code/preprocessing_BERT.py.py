import pandas as pd

df = pd.read_parquet('/home/ubuntu/NLP_Project_Team1/data/books_merged_clean.parquet')

# Sorting the dataframe
df_sorted = df.sort_values(by=['User_id', 'review/time'])

# Grouping by 'User_id' and aggregating 'Id' into lists
user_sequences = df_sorted.groupby('User_id')['Id'].apply(list)

# Convert Series to DataFrame
user_sequences_df = user_sequences.reset_index()
user_sequences_df.columns = ['User_id', 'Item_sequence']

# Print the DataFrame to check it
print(user_sequences_df.head())

# Save the DataFrame as a Parquet file
user_sequences_df.to_parquet('/home/ubuntu/NLP_Project_Team1/data/user_sequences.parquet')
