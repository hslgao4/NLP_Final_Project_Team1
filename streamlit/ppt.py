import streamlit as st

st.title("AI Book Recommendation")
st.subheader("Team 1: Caitlin Bailey, Nina Ebensperger & Liang  Gao")

st.divider()

st.header("Dataset")
st.image("/home/ubuntu/NLP_Project_Team1/streamlit/dataset1.png")

st.divider()
st.header("SVD Model")
st.subheader('Model Architecture')
st.image("/home/ubuntu/NLP_Project_Team1/streamlit/svd.png") #, width=600)
st.markdown('**Text preprocessing**: Cleaning and standardizing the text data to remove noise and irrelevant information.')
st.markdown('**TF-IDF vectorization**: Converting the text data into numerical vectors while emphasizing the importance of rare words in distinguishing documents.')
st.markdown('**SVD (Singular Value Decomposition**): Reducing the dimensionality of the TF-IDF matrix to capture latent semantic relationships.')
st.markdown('**LSA (Latent Semantic Analysis)**: Applying SVD to extract underlying topics or concepts from the document-term matrix.')
st.markdown('**Cosine similarity**: Calculating the similarity between documents based on their vector representations.')
st.divider()

st.subheader('Experiment')
st.image("/home/ubuntu/NLP_Project_Team1/streamlit/caitlin_1.png")
st.divider()

st.subheader('Result')
st.markdown('**Results of the qualitative analysis conducted through manual inspection of the model’s book recommendations.**')
st.image("/home/ubuntu/NLP_Project_Team1/streamlit/caitlin_2.png")
st.divider()


st.header("KNN Model")
st.markdown('First, we used a pre-trained BertForSequenceClassification model to predict the category based on the input review.')
st.markdown("Second, we filtered the dataframe to include only those data where the 'categories' column matches the predicted category.")
st.divider()

st.markdown('**Dataset precessing**')

st.divider()

st.markdown('**BertForSequenceClassification**')
st.markdown("*Tokenizatin*: Apply the Pretrained BertTokenizer, which is conten-based. Replace the classical tokenization methods, and don't need TFIDF which is frequency-based. ")
st.markdown("*Dataset*: Train 80%, Test 10% and Validation 10%")
st.divider()

st.markdown('**K-Nearest Neighbors(KNN)** works by finding the nearest neighbors to a query data point and then basing its prediction on the properties of these neighbors.')
st.markdown("KNN algorithm calculates the distance between a **user's new review text** and the **existing reviews** in our dataset.")
st.markdown('It then identifies and outputs the **indices** of the reviews closest to the input.')
st.markdown("We can effectively recommend books from our dataset that align with the user's preferences using these indices.")
st.divider()


st.divider()