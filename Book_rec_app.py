# To Run: streamlit run Book_rec_app.py --server.port=8888
import streamlit as st

import pickle
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")  # Download NLTK's tokenizers
nltk.download("wordnet")  # Download NLTK's WordNet lexical database
nltk.download("stopwords")  # Download NLTK's stopwords

#%%
st.set_page_config(
    page_title="AI Book Recommendation",
    page_icon=":books:",
)

#%%
st.title('AI Book Recommendation App')
with st.expander(":open_book: Welcome to AI Book Recommendation!", expanded=False):
    st.write(
        """     
    - Provide a short book review and receive tailored recommendations for your next read.
    - Use the intuitive controls on the left to tailor your experience to your preferences.
        """
    )

# Step-by-Step Guide
with st.expander(":open_book: How to Use", expanded=False):
    st.write(
        """
    1. **Enter Book Review:**
        - Type or paste a 3-5 sentence review of a book you enjoyed to find more like it.
        - Hint: Tell us what you liked most about a recent read.
        - Be descriptive, get creative! :stuck_out_tongue_winking_eye:

    2. **Choose Features:**
        - Toggle the switch on the left sidebar to choose between SVD and Transformers4Rec models.
        - Which model gives you better results?
        """
    )

#%%#
# SVD Model code

# Load the pickled model
with open('./data/SVD_recommendations.pkl', 'rb') as f:
    lsa_model = pickle.load(f)

# Load the pickled X (data)
with open('./data/X_matrix.pkl', 'rb') as f:
    X = pickle.load(f)

# Load the dataset
data = pd.read_parquet('./data/books_merged_clean.parquet')

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

# Function to clean Book title
def clean_title(title):
    # Lowercase all characters
    title = title.lower()
    # Remove comma and semicolon
    title = title.replace(',', '').replace(';', '')
    # Capitalize first word only
    title = title.capitalize()
    return title

# Function to generate recommendations
def get_recommendations(text_input, data, lsa_model, X):
    # Preprocess the text
    processed_text = preprocess_text(text_input)
    # Transform the preprocessed text input
    input_vector = lsa_model.transform([processed_text])
    # Compute cosine similarity between input vector and all book vectors
    similarities = cosine_similarity(input_vector, X)
    # Get indices of top recommendations
    top_indices = similarities.argsort()[0][::-1][:40]  # Get top 30 recommendations
    # Retrieve recommended books
    recommendations = data.iloc[top_indices]
    # Apply clean_title function to the 'Title' column
    recommendations = recommendations.copy() # ensure that any modifications are made to a separate copy
    recommendations.loc[:, 'Title'] = recommendations['Title'].apply(clean_title)
    # Remove duplicates based on review summary
    recommendations = recommendations.drop_duplicates(subset=['review/summary'], keep='first')
    # Sort recommendations by rating (review/score)
    recommendations = recommendations.sort_values(by='review/score', ascending=False)
    # Return top non-duplicate recommendations
    return recommendations.head(10)

# SVD main function
def svd_main():
    # Take user input via terminal
    input_text = input("For Future Reads recommendations, type your recent five-star book review here:")
    # Make recommendations based on the input text
    recommendations = get_recommendations(input_text, data, lsa_model, X)
    # Print recommendations
    print(recommendations[['Title', 'categories', 'review/summary']])

#%%
# Main function to run the Streamlit app
def main():
    st.title('Your next story starts here...')

    # Sidebar to select model type
    model_type = st.sidebar.radio("Select Model Type", ("SVD", "Transformers4Rec"))

    # Text input area for book review
    input_text = st.text_area("Enter your book review (3-5 sentences):")

    if st.button("Get Recommendations"):
        if model_type == "SVD":
            recommendations = svd_main(input_text)
            # st.write("SVD recommendations coming soon!")
            st.write("**Recommendations:**")
            st.table(recommendations)
        elif model_type == "Transformers4Rec":
            # recommendations = get_transformer4rec_recommendations(input_text, data)
            st.write("Transformers4Rec recommendations coming soon!")


        # Output recommendations
        # st.write("**Recommendations:**")
        # st.table(recommendations)


if __name__ == "__main__":
    main()