import pandas as pd
from transformers import BertTokenizer
from datasets import load_from_disk, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

dff = pd.read_csv('../Code/data/books_merged.csv')

'''Add columns here for your needs'''
df = dff[['Title', 'categories', 'description', 'authors', 'Id', 'review/text', 'User_id']]
df = df.copy()
df.rename(columns={'review/text': 'review_text'}, inplace=True)

# Filtering users with less than 3 reviews
def remove_users(data, user_column='User_id'):

    user_review_counts = data[user_column].value_counts()
    filtered_data = data[data[user_column].isin(user_review_counts[user_review_counts > 3].index)]
    filtered_user_count = filtered_data[user_column].nunique()
    print(f"Filtered user count: {filtered_user_count}")

    return filtered_data

df_temp = remove_users(df, user_column='User_id')

# Remove null value
data = df_temp.dropna().reset_index()

data.categories.value_counts().to_frame().reset_index()
data.loc[:, 'categories'] = data.loc[:, 'categories'].apply(lambda x: x.strip("[]'"))
data.loc[:, 'authors'] = data.loc[:, 'authors'].apply(lambda x: x.strip("[]'"))
category = data.categories.value_counts().to_frame().reset_index()

# Find categories with counts less than 2000 and set
category_less_than_1000 = category[category['count'] < 2000]
cate = category_less_than_1000['categories'].to_list()

# Drop count < 2000
df = data.copy()
df_filtered = df[~df['categories'].isin(cate)]
print("shape:", df_filtered.shape)
df_filtered = df_filtered.copy()

'''If you want to keep User_id, remove this from the next line of code'''
final_df = df_filtered.drop(columns=['index', 'User_id'], axis=1)



final_df.to_parquet('./Code/data/final_df.parquet')
'''final_df.shape (826840, 6)'''


'''Tokenization'''
df = pd.read_parquet('../Code/data/final_df.parquet')
print('shape:', df.shape)

'''I don't think you need this label I created for classification, but I think it's fine to run.'''
# Encode the category
label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['categories'])
# Save the label_encoder
joblib.dump(label_encoder, 'label_encoder.joblib')


# Split into train, test, and validation and save files (0.8, 0.1, 0.1)
train_df, temp_test_df = train_test_split(df, test_size=0.2, random_state=42)
test_df, valid_df = train_test_split(temp_test_df, test_size=0.5, random_state=42)


# Convert dataframes to Hugging Face dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(valid_df)
test_dataset = Dataset.from_pandas(test_df)

print(f'Train shape {train_df.shape}')
print(f'Test shape {test_df.shape}')
print(f'Validation shape {valid_df.shape}')
'''
Train shape (661472, 7)
Test shape (82684, 7)
Validation shape (82684, 7)
'''

# Tokenize with BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize(dataset):
    return tokenizer(dataset['review_text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

'''Save dataset to disk '''
data_path = './Code/train_data'
if not os.path.exists(data_path):
    os.makedirs(data_path)

train_dataset.save_to_disk('../Code/train_data/train_dataset')
val_dataset.save_to_disk('../Code/train_data/val_dataset')
test_dataset.save_to_disk('../Code/train_data/test_dataset')


'''Code to Load the dataset'''
train_dataset = load_from_disk('../Code/train_data/train_dataset')
val_dataset = load_from_disk('../Code/train_data/val_dataset')
test_dataset = load_from_disk('../Code/train_data/test_dataset')