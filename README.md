# AI Book Recs App (NLP_Project_Team1)
This project showcases work from DATS 6312: Natural Language Processing. Our interactie app showcases three different NLP models for book recommendation using data from the Amazon Books Reviews dataset. 

![Screenshot](images/Screenshot.png)

Here is a brief description of each model:

## Models Description

- **DeepLake**:
  - **Input**: Takes in book descriptions as input.
  - **Model**: Utilizes the SentenceTransformer model, specifically 'all-mpnet-base-v2', to transform textual descriptions into semantic vector embeddings.
  - **Functionality**: These embeddings are stored in VectorStore, part of the DeepLake system, which allows for efficient storage and quick retrieval of vector data. DeepLake facilitates rapid similarity searches using cosine similarity, enabling the system to find and recommend books that are semantically related to user queries.
  - **Advantages**: The integration of SentenceTransformer and VectorStore within DeepLake supports real-time data querying and highly relevant recommendation outputs, making it especially suitable for interactive applications like online bookstores where immediate response is critical.


- **KNN**: 
  - **Input**: 3-5 sentence book review
  - Pre-trained BertForSequenceClassification model predicts book category for user based on input review
  - KNN finds nearest books in the predicted category using the input review 

- **SVD**: 
  - **Input**: 3-5 sentence book review
  - Model uses dimensionality reduction and cosine similary to identify similar book reviews
  - Architecture uses NLTK preprocessing, TF-IDF vectorization, SVD, LSA, and cosine similarity

## Getting Started

- Clone the repository
- Install dependencies
- See README file under 'code' folder for instructions about individual model setup
- Run `Book_rec_app_final.py` to start the app

## Built With

- [Streamlit](https://streamlit.io/) - For the web interface
- [Amazon Books Reviews Data](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews) - For training and fine-tuning models
- [DeepLake](https://docs.activeloop.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme) - For DeepLake NLP model

## Contributors

- Caitlin Bailey
- Nina Ebensperger
- Liang Gao


