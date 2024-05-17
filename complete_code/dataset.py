import pandas as pd

dff = pd.read_csv('data/books_merged.csv')
df = dff[['Title', 'categories', 'description', 'authors', 'Id', 'review/text', 'User_id', 'review/score']]
df = df.copy()
df.rename(columns={'review/text': 'review_text'}, inplace=True)

# Filtering users with only one review
def rem_once_review(data, user_column='User_id'):
    user_review_counts = data[user_column].value_counts()
    filtered_data = data[data[user_column].isin(user_review_counts[user_review_counts > 10].index)]
    filtered_user_count = filtered_data[user_column].nunique()
    print(f"Filtered user count: {filtered_user_count}")

    return filtered_data

df_temp = rem_once_review(df, user_column='User_id')

# Remove null value
data = df_temp.dropna().reset_index()

data.categories.value_counts().to_frame().reset_index()
data.loc[:, 'categories'] = data.loc[:, 'categories'].apply(lambda x: x.strip("[]'"))
data.loc[:, 'authors'] = data.loc[:, 'authors'].apply(lambda x: x.strip("[]'"))
category = data.categories.value_counts().to_frame().reset_index()

# Find categories with counts less than 5000 and set
delete_category = category[(category['count'] < 5000) | (category['count'] > 20000)]

cate = delete_category['categories'].to_list()


df = data.copy()
df_filtered = df[~df['categories'].isin(cate)]
print("shape:", df_filtered.shape)

df_filtered = df_filtered.copy()
final_df = df_filtered.drop(columns=['index', 'User_id'], axis=1)

final_df.to_parquet('data/final_df.parquet')

'''final_df.shape (58199, 6)'''