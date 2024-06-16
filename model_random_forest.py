from sklearn.ensemble import RandomForestClassifier
import streamlit as st


def train_random_forest(X_train, y_train, X_test, y_test): #
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    st.write(f"<p style='font-size:24px'>{model.score(X_test, y_test)}</p>", unsafe_allow_html=True)
    return model