from sklearn.ensemble import GradientBoostingClassifier
import streamlit as st

def train_gradient_boosting(X_train, y_train, X_test, y_test):

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    st.write(f"<p style='font-size:24px'>{model.score(X_test, y_test)}</p>", unsafe_allow_html=True)
    return model