import streamlit as st
import nltk
import sklearn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import os
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_file_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def run_ml_app():
  st.subheader('Fake News Classification')
  st.write("This model uses Lemmatization as Data Preprocessing, TF-IDF as Feature Engineering and XGBoost for Accuracy.")
  input_title = st.text_area("Insert news title")
  input_content = st.text_area("Insert news content")
  if st.button("Classify"):
    # combine
    combined_title_content = input_title + " " + input_content
    # tokenize
    tokenized_news = word_tokenize(combined_title_content)
    # stop words
    stop_words = set(stopwords.words('english'))
    filtered_news = [word for word in tokenized_news if word.lower() not in stop_words]
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_news = [lemmatizer.lemmatize(token) for token in filtered_news]
    # joining into text
    lemmatized_text = ' '.join(lemmatized_news)
    # vectorizer
    vectorizer = load_file_pickle('vectorizer.pkl')
    ready_news = vectorizer.transform([lemmatized_text])
    # Prediction
    st.subheader('Prediction Result')
    model = load_file_pickle('model_xgboost.pkl')

    prediction = model.predict(ready_news)
    pred_proba = model.predict_proba(ready_news)
    
    pred_probability_score = {'Real News percentage':round(pred_proba[0][1]*100,4),
                                'Fake News percentage':round(pred_proba[0][0]*100,4)}

    if prediction == 1:
      st.success("It's real news")
      st.write(pred_probability_score)
    else:
      st.warning("It's fake news")
      st.write(pred_probability_score)
