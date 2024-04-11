import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse import save_npz

def create_sparse_user_item_matrix(data, user_column='User_id', item_column='Id', interaction_column='review/score'):
    """Create a sparse user-item interaction matrix."""
    

    # Filter out rows where 'User_id' or 'Id' is None
    data = data.dropna(subset=[user_column, item_column])
# Map user and item IDs to integer indices
    user_ids = data[user_column].astype("category").cat.codes
    item_ids = data[item_column].astype("category").cat.codes

    # Example debugging code
    print("Min user category code:", user_ids.min())
    print("Min item category code:", item_ids.min())

    if (user_ids.min() < 0) or (item_ids.min() < 0):
        print("Negative category code found!")
        # Identify rows with issues
        print("Issue in user_ids:", data.loc[user_ids < 0, 'User_id'])
        print("Issue in item_ids:", data.loc[item_ids < 0, 'Id'])

    
    # Create a sparse matrix
    interaction_matrix = coo_matrix((data[interaction_column].fillna(0), (user_ids, item_ids)), 
                                    shape=(data[user_column].nunique(), data[item_column].nunique()))
    
    return interaction_matrix, data[user_column].astype("category"), data[item_column].astype("category")

def main():
    # Existing code to load data and create the matrix
    data_path = "/home/ubuntu/NLP_Project_Team1/data/books_merged_clean.parquet"
    data = pd.read_parquet(data_path)
    # ... your data processing steps

    # Create sparse user-item matrix
    interaction_matrix, user_categories, item_categories = create_sparse_user_item_matrix(data)

    # Save the interaction matrix
    save_npz("/home/ubuntu/NLP_Project_Team1/data/user_item_interaction_matrix.npz", interaction_matrix)

    print("Interaction matrix saved.")

if __name__ == "__main__":
    main()
