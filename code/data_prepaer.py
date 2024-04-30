import pandas as pd

dff = pd.read_csv('../Code/data/books_merged.csv')
df = dff[['Title', 'categories', 'description', 'authors', 'Id', 'review/text', 'User_id', 'review/summary', 'review/score']]
df = df.copy()
df.rename(columns={'review/text': 'review_text', 'review/summary': 'review_summary', 'review/score': 'review_score'}, inplace=True)

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

final_df.to_parquet('../Code/data/final_df.parquet')

'''final_df.shape (58199, 6)'''

'''
             categories  count
0              Religion  18894
1  Business & Economics  10813
2   Young Adult Fiction  10582
3        Social Science   6624
4            Philosophy   6131
5               Science   5155
'''

# # Manually set those less than 3000 to others
# category_less_than_1000 = category[category['count'] < 3000]
# cat = category_less_than_1000['categories'].to_list()
#
# data = df_filtered.copy()
# data['new_categories'] = data['categories']
# final_df = data.copy()
# final_df.loc[final_df['new_categories'].isin(cat), 'new_categories'] = 'others'
# final_df = final_df.reset_index()
# final_df = final_df.drop(columns=['level_0', 'index', 'categories'], axis=1)
# final_df.to_parquet('./Code/data/final_df.parquet')




# train_df.to_csv('data/train_df.csv', index=False)
# test_df.to_csv('data/test_df.csv', index=False)
# valid_df.to_csv('data/valid_df.csv', index=False)
# dataset = load_dataset('csv', data_files={'train': '../Code/data/train_df.csv',
#                                           'validation': '../Code/data/valid_df.csv',
#                                           'test': '../Code/data/test_df.csv'})

