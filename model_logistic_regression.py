from sklearn.linear_model import LogisticRegression
import streamlit as st
import pandas as pd



def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    st.write(f"<p style='font-size:24px'>{model.score(X_test, y_test)}</p>", unsafe_allow_html=True)
    return model
    
