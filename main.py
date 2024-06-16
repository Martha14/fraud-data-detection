import streamlit as st
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from model_random_forest import train_random_forest
from model_logistic_regression import train_logistic_regression
from model_neural_network import train_neural_network
from model_gradient_boosting import train_gradient_boosting
import seaborn as sns


def main():
    st.title("Fraud data detection")
    data = pd.read_csv('creditcard.csv')
    data['Amount'] = RobustScaler().fit_transform(data['Amount'].to_numpy().reshape(-1,1))
    data['Time'] = (data['Time']- data['Time'].min()) / (data['Time'].max() - data['Time'].min())
    X = data.drop('Class', axis=1)
    y = data['Class']   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    df_majority = data[data['Class']==0]
    df_minority = data[data['Class']==1]

    df_minority_upsampled = resample(df_minority, 
                                    replace=True,     # sample with replacement
                                    n_samples=284315,    # to match majority class
                                    random_state=123) # reproducible results
    
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    

    X_res = df_upsampled.drop('Class', axis=1)
    y_res = df_upsampled['Class']

    X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    model_choice = st.sidebar.selectbox("Choose an option:", ["---", "View data", "Logistic Regression", "Random Forest", "Gradient Boosting", "Neural Network"])

    if model_choice == "---":
        st.write("""
                 Welcome to the Fraud data detection app. 
                 To start, choose an option from the menu on the left.
                 """)  
        
    if model_choice == "View data":
        st.sidebar.write('''You can view the data here. As you cas notice, the data is highly imbalanced. 
                         The number of fraud cases is much lower than the number of non-fraud cases.
                         We will need to address this issue before training the model.
                         ''')
        st.write("Data:")
        st.dataframe(data)
        st.write("Statistics:")
        st.write(data.describe())
        st.write("Class distribution:")
        st.write(data['Class'].value_counts())
    
        fig, ax = plt.subplots(figsize=(85, 85))
        data.hist(bins=30, ax=ax)
        st.pyplot(fig)

    if model_choice == "Logistic Regression":
        st.sidebar.write('''You can train the model using Logistic Regression here.
                            We will train the model on the original data and on the upsampled data to compare the results.
                         First result is for the original data, second for the upsampled data.
                            ''')
        st.write("Logistic Regression scores. As you can see, the results are different for the original and upsampled data. But which one is better?")
        with st.spinner('Training the model...'):
            model_lr = train_logistic_regression(X_train, y_train, X_test, y_test)
            model_lr_res = train_logistic_regression(X_train_res, y_train_res, X_test_res, y_test_res)
        y_pred = model_lr.predict(X_test)
        y_pred_res = model_lr_res.predict(X_test_res)
        cm = confusion_matrix(y_test, y_pred)
        cm_res = confusion_matrix(y_test_res, y_pred_res)
        st.write("Confusion Matrix compare")
        st.write(cm, cm_res)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0])
        ax[0].set_title("Confusion Matrix")
        ax[0].set_xlabel("Predicted")
        ax[0].set_ylabel("Actual")
        sns.heatmap(cm_res, annot=True, fmt="d", cmap="Blues", ax=ax[1])
        ax[1].set_title("Confusion Matrix Resampled")
        ax[1].set_xlabel("Predicted")
        ax[1].set_ylabel("Actual")
        st.write("Confusion Matrix Comparison")
        st.pyplot(fig)

        report = classification_report(y_test, y_pred, output_dict=True, target_names=['Non-fraud', 'Fraud'])
        report_res = classification_report(y_test_res, y_pred_res, output_dict=True, target_names=['Non-fraud', 'Fraud'])
        report_df = pd.DataFrame(report).transpose()
        report_df_res = pd.DataFrame(report_res).transpose()
        st.write("Classification Report compare")
        st.table(report_df)
        st.table(report_df_res)

    if model_choice == "Random Forest":
        st.sidebar.write('''You can train the model using Random Forest here.
                            We will train the model on the original data and on the upsampled data to compare the results.
                            First result is for the original data, second for the upsampled data.
                            ''')
        with st.spinner('Training the model...'):
            model_rf = train_random_forest(X_train, y_train, X_test, y_test)
            model_rf_res = train_random_forest(X_train_res, y_train_res, X_test_res, y_test_res)
        st.write('Random Forest scores. As you can see, the results are different for the original and upsampled data. But which one is better?')
        y_pred = model_rf.predict(X_test)
        y_pred_res = model_rf_res.predict(X_test_res)
        cm = confusion_matrix(y_test, y_pred)
        cm_res = confusion_matrix(y_test_res, y_pred_res)
        st.write("Confusion Matrix compare")
        st.write(cm, cm_res)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0])
        ax[0].set_title("Confusion Matrix")
        ax[0].set_xlabel("Predicted")
        ax[0].set_ylabel("Actual")
        sns.heatmap(cm_res, annot=True, fmt="d", cmap="Blues", ax=ax[1])
        ax[1].set_title("Confusion Matrix Resampled")
        ax[1].set_xlabel("Predicted")
        ax[1].set_ylabel("Actual")
        st.write("Confusion Matrix Comparison")
        st.pyplot(fig)

        report = classification_report(y_test, y_pred, output_dict=True, target_names=['Non-fraud', 'Fraud'])
        report_res = classification_report(y_test_res, y_pred_res, output_dict=True, target_names=['Non-fraud', 'Fraud'])
        report_df = pd.DataFrame(report).transpose()
        report_df_res = pd.DataFrame(report_res).transpose()
        st.write("Classification Report compare")
        st.table(report_df)
        st.table(report_df_res)
    
    if model_choice == "Gradient Boosting":
        st.sidebar.write('''You can train the model using Gradien Boosting here.
                            We will train the model on the original data and on the upsampled data to compare the results.
                            First result is for the original data, second for the upsampled data.
                            ''')
        with st.spinner('Training the model...'):
            model_gb = train_gradient_boosting(X_train, y_train, X_test, y_test)
            model_gb_res = train_gradient_boosting(X_train_res, y_train_res, X_test_res, y_test_res)
        st.write('Gradien Boosting scores. As you can see, the results are different for the original and upsampled data. But which one is better?')
        y_pred = model_gb.predict(X_test)
        y_pred_res = model_gb_res.predict(X_test_res)
        cm = confusion_matrix(y_test, y_pred)
        cm_res = confusion_matrix(y_test_res, y_pred_res)
        st.write("Confusion Matrix compare")
        st.write(cm, cm_res)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0])
        ax[0].set_title("Confusion Matrix")
        ax[0].set_xlabel("Predicted")
        ax[0].set_ylabel("Actual")
        sns.heatmap(cm_res, annot=True, fmt="d", cmap="Blues", ax=ax[1])
        ax[1].set_title("Confusion Matrix Resampled")
        ax[1].set_xlabel("Predicted")
        ax[1].set_ylabel("Actual")
        st.write("Confusion Matrix Comparison")
        st.pyplot(fig)

        report = classification_report(y_test, y_pred, output_dict=True, target_names=['Non-fraud', 'Fraud'])
        report_res = classification_report(y_test_res, y_pred_res, output_dict=True, target_names=['Non-fraud', 'Fraud'])
        report_df = pd.DataFrame(report).transpose()
        report_df_res = pd.DataFrame(report_res).transpose()
        st.write("Classification Report compare")
        st.table(report_df)
        st.table(report_df_res)

    if model_choice == "Neural Network":
        st.sidebar.write('''You can train the model using Neural Network here.
                            We will train the model on the original data and on the upsampled data to compare the results.
                            First result is for the original data, second for the upsampled data.
                            ''')
        with st.spinner('Training the model...'):
            model_nn = train_neural_network(X_train, y_train, X_test, y_test)
            model_nn_res = train_neural_network(X_train_res, y_train_res, X_test_res, y_test_res)
        st.write('Neural Network scores. As you can see, the results are different for the original and upsampled data. But which one is better?')
        y_pred = model_nn.predict(X_test).argmax(axis=1)
        y_pred_res = model_nn_res.predict(X_test_res).argmax(axis=1)
        cm = confusion_matrix(y_test, y_pred)
        cm_res = confusion_matrix(y_test_res, y_pred_res)
        st.write("Confusion Matrix compare")
        st.write(cm, cm_res)
        report = classification_report(y_test, y_pred, output_dict=True, target_names=['Non-fraud', 'Fraud'],)
        report_res = classification_report(y_test_res, y_pred_res, output_dict=True, target_names=['Non-fraud', 'Fraud'])
        report_df = pd.DataFrame(report).transpose()
        report_df_res = pd.DataFrame(report_res).transpose()
        st.write("Classification Report compare")
        st.table(report_df)
        st.table(report_df_res)

if __name__ == "__main__":
    main()


# porównanie działania modeli (wyniki, czas wykonania, itp.) na oryginalnych danych i danych po oversamplingu
# zapisanie wyników do pliku
# zapisanie modelu do pliku
# porównanie wszystkich modeli ze sobą 


