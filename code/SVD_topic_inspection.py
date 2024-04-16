import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from gensim.models import CoherenceModel
from nltk.tokenize import word_tokenize
from gensim.corpora.dictionary import Dictionary

# Function to preprocess the text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespaces
    text = " ".join(text.split())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    text = " ".join([word for word in tokens if word not in stop_words])
    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    text = " ".join([lemmatizer.lemmatize(word) for word in tokens])
    return text

# Function to load and clean the data
def load_and_clean_dataset(file_path, summary_column='review/summary', text_column='review/text'):
    # Load the dataset from Parquet file
    data = pd.read_parquet(file_path)
    # Fill missing values for "review/title" and "review/summary" columns with empty strings
    data.fillna({'review/title': '', 'review/summary': ''}, inplace=True)
    # Combine text data
    data['review_combined'] = data[summary_column] + ' ' + data[text_column]
    # Apply text preprocessing
    data['review_combined'] = data['review_combined'].apply(preprocess_text)
    # Drop rows with missing values for specified columns
    columns_to_check = ['review_combined']
    cleaned_data = data.dropna(subset=columns_to_check, how='any')
    return cleaned_data

# Load and clean your dataset
data = load_and_clean_dataset('/home/ubuntu/caitlin/NLP_Project_Team1/data/books_merged_clean.parquet')

# Extract text from the 'review_combined' column
documents = data['review_combined'].tolist()

# Tokenize documents
tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]

# Create dictionary
dictionary = Dictionary(tokenized_documents)

# Load LSA model with 17 components
with open('SVD_recommendations_17.pkl', 'rb') as f:
    svd_model = pickle.load(f)
print(type(svd_model))

# Load X matrix
with open('X_matrix_17.pkl', 'rb') as f:
    X = pickle.load(f)
print(type(X))

# Access the TruncatedSVD model from the Pipeline
svd_model = svd_model.named_steps['truncatedsvd']

# Get the most important words for each topic
topic_terms = []
for i, component in enumerate(svd_model.components_):
    # Get top words based on X matrix
    top_word_indices = component.argsort()[:-11:-1]
    topic_terms.append([dictionary[ind] for ind in top_word_indices])

# Compute coherence score
coherence_model = CoherenceModel(topics=topic_terms, texts=tokenized_documents, dictionary=dictionary, coherence='c_v',
                                 model=svd_model, corpus=X)
coherence_score = coherence_model.get_coherence()
print("Coherence Score:", coherence_score)
