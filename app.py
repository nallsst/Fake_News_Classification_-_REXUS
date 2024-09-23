import streamlit as st
import streamlit.components.v1 as stc
from ml_app import run_ml_app

def main():
    menu = ['Home', 'Machine Learning']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.subheader('Welcome to Rexus Final Project')
        st.write("Click the selecbox on the left")
        st.write("And then choose 'Machine Learning' to start Classification  on Fake News")
        st.write("Here is the link of Fake News Dataset: https://www.kaggle.com/datasets/rajatkumar30/fake-news")
    elif choice == 'Machine Learning':
        #st.subheader('Welcome to Our Machine Learning')
        run_ml_app()

        
if __name__ == '__main__':
    main()