# AI Book Recs App (NLP_Project_Team1)
This project showcases work from DATS 6312: Natural Language Processing. Our interactie app showcases three different NLP models for book recommendation using data from the Amazon Books Reviews dataset. 

![Screenshot](images/Screenshot.png)

Here is a brief description of each model:

## Models

- **DeepLake**: 
  - ADD INFO HERE

- **KNN**: 
  - Input a 3-5 sentence book review
  - Pre-trained BertForSequenceClassification model predicts book category for user based on input review
  - KNN finds nearest books in the predicted category using the input review 

- **SVD**: 
  - Input a 3-5 sentence book review
  - Model uses dimensionality reduction and cosine similary to identify similar book reviews
  - Architecture uses NLTK preprocessing, TF-IDF vectorization, SVD, LSA, and cosine similarity

## Getting Started

- Clone the repository
- Install dependencies
- Run `Book_rec_app_final.py` to start the app

## Built With

- [Streamlit](https://streamlit.io/) - For the web interface
- [Amazon Books Reviews Data](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews) - For training and fine-tuning models
- [DeepLake](https://docs.activeloop.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme) - For DeepLake NLP model

## Contributors

- Caitlin Bailey
- Nina Ebensperger
- Liang Gao


