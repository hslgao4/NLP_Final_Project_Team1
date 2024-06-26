import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import joblib
from sklearn.neighbors import NearestNeighbors
import numpy as np
import nltk

nltk.download('punkt')
nltk.download('wordnet')
import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from deeplake import VectorStore
from sentence_transformers import SentenceTransformer
import pandas as pd

tabs = st.tabs(["📖 My Slides", "💻 My Demo"])

with tabs[0]:
    import streamlit as st

    st.title("AI Book Recommendation")
    st.subheader("Team 1: Caitlin Bailey, Nina Ebensperger & Liang  Gao")

    st.divider()

    st.header("Dataset")
    st.image("/home/ubuntu/NLP_Project_Team1/streamlit/dataset1.png")

    st.divider()
    st.header("SVD Model")
    st.subheader('Model Architecture')
    st.image("/home/ubuntu/NLP_Project_Team1/streamlit/svd.png")  # , width=600)
    st.markdown(
        '**Text preprocessing**: Cleaning and standardizing the text data to remove noise and irrelevant information.')
    st.markdown(
        '**TF-IDF vectorization**: Converting the text data into numerical vectors while emphasizing the importance of rare words in distinguishing documents.')
    st.markdown(
        '**SVD (Singular Value Decomposition**): Reducing the dimensionality of the TF-IDF matrix to capture latent semantic relationships.')
    st.markdown(
        '**LSA (Latent Semantic Analysis)**: Applying SVD to extract underlying topics or concepts from the document-term matrix.')
    st.markdown(
        '**Cosine similarity**: Calculating the similarity between documents based on their vector representations.')
    st.divider()

    st.subheader('Experiment')
    st.image("/home/ubuntu/NLP_Project_Team1/streamlit/caitlin_1.png")
    st.divider()

    st.subheader('Result')
    st.markdown(
        '**Results of the qualitative analysis conducted through manual inspection of the model’s book recommendations.**')
    st.image("/home/ubuntu/NLP_Project_Team1/streamlit/caitlin_2.png")
    st.divider()

    st.header("KNN Model")
    st.markdown('''
        * **First**, we used a pre-trained **BertForSequenceClassification** model to predict the **category** based on the input review.
        * **Second**, we filtered the dataframe to include only those data where the 'categories' column matches the predicted category and use **KNN** to recommend.)
        ''')
    st.divider()

    st.markdown('**Dataset preprocessing**')
    st.markdown('''
    * The observations of raw data is 3 million, about **3GB**. **Lack of GPU capacity**.
    * Dropped rows with **null value**.
    * Dropped categories with a count of **less than 5,000 and greater than 20,000**.
    * After filtering, the dataset for later modeling has **58,199 observations**.
    ''')
    st.divider()

    st.markdown('**BertForSequenceClassification**')
    st.markdown('''
    * Tokenizatin: Apply the Pretrained BertTokenizer. 
    * LabelEncoder: Encoded the book categories.
    * Dataset: Train 80%, Test 10% and Validation 10%.
    
    ''')

    st.markdown('**K-Nearst-Neighbors KNN**')
    st.markdown('''
    * **Filtered** the dataframe to include only data match that predicted category.
    * KNN calculates the distance between a **user's new review text** and the **existing reviews** in our dataset.
    * We can effectively recommend books from our dataset that align with the user's preferences using these indices.
    ''')
    st.divider()

    st.markdown('**Post processing**')
    st.markdown('''
    * Dropped rows that have the **same review summary**.
    * Dropped duplicate book that have the **same title**.
    ''')
    st.image("/home/ubuntu/NLP_Project_Team1/streamlit/liang.png")
    st.markdown('''* **Rank books based on review score** and recommend the top 6.''')

    st.divider()

    st.markdown('**Results**')
    st.markdown('''
    * Bert classifier Test results: F1-micro at 0.84, F1-Macro at 0.83, Cohen Kappa score at 0.80.''')
    st.markdown('''* Manul assessment result: ''')
    st.image("/home/ubuntu/NLP_Project_Team1/streamlit/liang_result.png", width=400)



    st.divider()
    st.header("DeepLake Sentence Transformer")
    st.subheader('Model Architecture')
    st.image("/home/ubuntu/NLP_Project_Team1/streamlit/nina_1.png")

    st.subheader('DeepLake Model Architecture')
    st.image("/home/ubuntu/NLP_Project_Team1/streamlit/nina_2.png")

    st.divider()

    st.subheader('Vector Store')
    st.markdown('**A specialized storage solution for handling vector embeddings.**')
    st.markdown('**Optimizes retrieval operations using embeddings for similarity searches.**')

    st.markdown('**Integration with Sentence Transformers**')
    st.markdown('''
    * Utilizes pre-trained Sentence Transformer models to generate embeddings.
    * Converts text data into vector embeddings for efficient similarity matching.
    ''')

    st.markdown('**Application in Book Recommendation**')
    st.markdown('''
    * Stores book descriptions as embeddings.
    * Facilitates finding books with similar themes using cosine similarity searches.
    ''')


    st.markdown('**Advantages of VectorStore**')
    st.markdown('''
    * Fast retrieval times optimized for high-dimensional data.
    * Scalable and adaptable to various types of data including text and images.
    ''')








with tabs[1]:
    st.title('AI Book Recommendation App')
    with st.expander(":open_book: Welcome to AI Book Recommendation!", expanded=False):
        st.write(
            """     
        - Provide a short book review or tell us what type of story you want to read
        - Receive tailored recommendations for your next read
        - Use the intuitive controls on the left to tailor your experience to your preferences.
            """
        )

    # Step-by-Step Guide
    with st.expander(":open_book: How to Use", expanded=False):
        st.write(
            """
        1. **Enter Book Summary (DeepLake model):**
            - Type or paste a description or summary of the type of book you want to read.
            - Be descriptive, get creative! :stuck_out_tongue_winking_eye:

        2. **Enter Book Review (KNN & SVD models):**
            - Type or paste a 3-5 sentence review of a book you enjoyed to find more like it.
            - Hint: Tell us what you liked most about a recent read.

        3. **Choose Features:**
            - Toggle the switch on the left sidebar to choose between models.
            - Which model gives you better results?
            """
        )


    # %%#

    # DeepLake
    def load_and_clean_dataset(file_path):
        print("Loading data...")
        data = pd.read_parquet(file_path)
        print("Data loaded.")

        # Standardize titles and authors by capitalizing the first letter of each word
        data['Title'] = data['Title'].str.title().str.strip()
        data['authors'] = data['authors'].str.title().str.strip()

        # Group by book identifiers (excluding 'Id') and select the entry with the highest review score
        data = data.groupby(['Title', 'authors']).agg({
            'review/score': 'max',  # Gets the maximum review score for each group
            'description': 'first'  # Keeps the first description found for each group
        }).reset_index()

        print("Data cleaned and filtered to include only the highest reviewed entries.")
        return data


    def create_vector_store(data, model, vector_store_path="my_vector_store", batch_size=500):
        print("Creating VectorStore...")
        vector_store = VectorStore(path=vector_store_path, overwrite=True)
        total_batches = len(data) // batch_size + (len(data) % batch_size != 0)

        seen_titles = set()
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i + batch_size]
            batch = batch[~batch['Title'].isin(seen_titles)]
            seen_titles.update(batch['Title'].tolist())

            texts = batch['description'].tolist()  # Using book descriptions for embedding
            embeddings = model.encode(texts, show_progress_bar=False)
            metadata = batch[['Title', 'authors', 'review/score']].to_dict(orient='records')
            vector_store.add(embedding=embeddings, metadata=metadata, text=texts)
            print(f"Processed batch {i // batch_size + 1}/{total_batches}")
        vector_store.commit()
        print("VectorStore created and data added.")
        return vector_store


    def get_deeplake_recs(input_text, vector_store, model, top_n=10):
        print("Finding similar books...")
        input_embedding = model.encode([input_text])[0]
        search_results = vector_store.search(embedding=input_embedding, k=top_n, distance_metric='COS')
        if search_results:
            similar_books = pd.DataFrame(search_results['metadata'])
            similar_books['Title'] = similar_books['Title'].str.title()
            similar_books['authors'] = similar_books['authors'].str.title()
            print("Most similar books found:")
            print(similar_books[['Title', 'authors', 'review/score']].head(6))
        else:
            print("No similar books found.")
            similar_books = pd.DataFrame()
        return similar_books


    # KNN
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)


    def model_predict(text):
        path1 = "/home/ubuntu/NLP_Project_Team1/Code/result/checkpoint-4656"
        model = BertForSequenceClassification.from_pretrained(path1)
        encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        model.eval()
        model.to(device)
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            output = model(**encoded_input)

        logits = output.logits
        predicted_class_idx = torch.argmax(logits, dim=1).item()
        label_encoder = joblib.load('/home/ubuntu/NLP_Project_Team1/label_encoder.joblib')
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])
        category = predicted_class[0]
        return category


    def generate_embedding(text):
        tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        tokens = {key: val.to(device) for key, val in tokens.items()}
        with torch.no_grad():
            outputs = bert_model(**tokens)

        return outputs.last_hidden_state.mean(dim=1).cpu().numpy().reshape(-1)


    def rem_deplicate(text):
        clean_text = re.sub(r"\[.*?\]", "", text)
        return re.sub(r"\(.*?\)", "", clean_text)


    def knn_model(text):
        path2 = '/home/ubuntu/NLP_Project_Team1/Code/data/final_df.parquet'
        dff = pd.read_parquet(path2)

        category = model_predict(text)
        df = dff.copy()
        df = df[df['categories'] == category].reset_index(drop=True)

        df['review_embed'] = df['review_text'].apply(generate_embedding)

        knn_model = NearestNeighbors(n_neighbors=50)
        knn_model.fit(np.stack(df['review_embed'].values))

        new_embedding = generate_embedding(text)
        distances, indices = knn_model.kneighbors([new_embedding])
        recommended_books = df.iloc[indices[0]]
        recommended_books = recommended_books.drop_duplicates(subset=['review_summary'], keep='first')
        books = recommended_books[['Title', 'authors', 'review_summary', 'review_score']]
        books = books.copy()
        books.Title = books.Title.apply(rem_deplicate)
        book_unique = books.drop_duplicates(subset=['Title']).copy()
        book_unique = book_unique.sort_values(by='review_score', ascending=False)
        temp = book_unique.copy()
        top6_book = temp.head(6)
        return top6_book


    # SVD Model code
    @st.cache_resource()
    # Lazy load the model and data
    def lazy_load_model_and_data():
        with open('/home/ubuntu/NLP_Project_Team1/data/SVD_recommendations_17.pkl', 'rb') as f:
            lsa_model = pickle.load(f)
        with open('/home/ubuntu/NLP_Project_Team1/data/X_matrix_17.pkl', 'rb') as f:
            X = pickle.load(f)
        data = pd.read_parquet('/home/ubuntu/NLP_Project_Team1/data/books_merged_clean.parquet')
        return lsa_model, X, data


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
    def get_svd_recommendations(text_input):
        # Lazy load model and data
        lsa_model, X, data = lazy_load_model_and_data()
        # Preprocess the text
        processed_text = preprocess_text(text_input)
        # Transform the preprocessed text input
        input_vector = lsa_model.transform([processed_text])
        # Compute cosine similarity between input vector and all book vectors
        similarities = cosine_similarity(input_vector, X)
        # Get indices of top recommendations
        top_indices = similarities.argsort()[0][::-1][:35]  # Get top 35 recommendations
        # Retrieve recommended books
        recommendations = data.iloc[top_indices]
        # Apply clean_title function to the 'Title' column
        recommendations = recommendations.copy()  # ensure that any modifications are made to a separate copy
        recommendations.loc[:, 'Title'] = recommendations['Title'].apply(clean_title)
        # Convert 'review/summary' and 'Title' columns to lowercase for duplicate removal
        recommendations['review/summary_lower'] = recommendations['review/summary'].str.lower()
        recommendations['Title_lower'] = recommendations['Title'].str.lower()
        # Remove duplicates based on review summary
        recommendations = recommendations.drop_duplicates(subset=['review/summary_lower'], keep='first')
        # Remove duplicates based on Title
        recommendations = recommendations.drop_duplicates(subset=['Title_lower'], keep='first')
        # Sort recommendations by rating (review/score)
        recommendations = recommendations.sort_values(by='review/score', ascending=False)
        # Return top non-duplicate recommendations
        return recommendations[['Title', 'authors', 'review/summary', 'review/score']].head(6)


    # %%
    # Main function to run the Streamlit app
    def main():
        st.title('Your next story starts here...')

        # Sidebar to select model type
        model_type = st.sidebar.radio("Select Model Type", ("DeepLake", "KNN", "SVD"))

        # Text input area for book review
        input_text = st.text_area("Enter your book description (DeepLake) or review (KNN, SVD):")

        if st.button("Get Recommendations"):
            if model_type == "DeepLake":
                data = load_and_clean_dataset("/home/ubuntu/NLP_Project_Team1/data/final_df.parquet")
                model = SentenceTransformer('all-mpnet-base-v2')
                vector_store_path = "/home/ubuntu/NLP_Project_Team1/my_vector_store/"
                vector_store = VectorStore(path=vector_store_path)  # Load the existing VectorStore
                recommendations = get_deeplake_recs(input_text, vector_store, model, top_n=6)
                st.write("**Recommendations:**")
                st.dataframe(recommendations, hide_index=True)
            elif model_type == "KNN":
                st.write("**Recommendations:**")
                recommendation = knn_model(input_text)
                st.dataframe(recommendation, hide_index=True)
            elif model_type == "SVD":
                recommendations = get_svd_recommendations(input_text)
                st.write("**Recommendations:**")
                st.dataframe(recommendations, hide_index=True)


    if __name__ == "__main__":
        main()