import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Function to load and preprocess the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    label_encoders = {}
    for col in ['Sex', 'BP', 'Cholesterol', 'Drug']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    return data, label_encoders

# Function to train the model and generate predictions
def train_and_evaluate(data):
    X = data.drop('Drug', axis=1)
    y = data['Drug']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, classification_report(y_test, y_pred, target_names=label_encoders['Drug'].classes_)

# Streamlit app
st.title("Drug Classification using Naive Bayes")

# Upload data file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Load and preprocess data
    data, label_encoders = load_data(uploaded_file)
    st.write("### Data Preview")
    st.write(data.head())

    # Train the model and display the results
    accuracy, classification_rep = train_and_evaluate(data)
    st.write("### Model Accuracy")
    st.write(accuracy)
    
    st.write("### Classification Report")
    st.text(classification_rep)
