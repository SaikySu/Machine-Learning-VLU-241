import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

st.set_page_config(page_title="Wine Classification App", layout="wide")
st.title("Wine Classification using KNN")

# Sidebar để điều chỉnh tham số
st.sidebar.header("Model Parameters")
k_value = st.sidebar.slider("Select number of neighbors (k)", 1, 20, 5)
test_size = st.sidebar.slider("Select test size ratio", 0.1, 0.5, 0.3)
random_state = st.sidebar.number_input("Set random state", value=42)

# Load dữ liệu
@st.cache_data  
def load_data():
    wine_data = load_wine()
    df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    df['target'] = wine_data.target
    return df, wine_data.target_names

df, target_names = load_data()

# Hiển thị dữ liệu thô
st.subheader("Raw Wine Dataset")
st.dataframe(df)

# Thống kê cơ bản về dataset
st.subheader("Dataset Statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Samples", df.shape[0])
with col2:
    st.metric("Features", df.shape[1] - 1)
with col3:
    st.metric("Classes", len(target_names))

# Train model và hiển thị kết quả
st.subheader("Model Training and Evaluation")

if st.button("Train Model"):
    # Chia dữ liệu
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Training progress
    with st.spinner('Training model...'):
        # Khởi tạo và train model
        knn = KNeighborsClassifier(n_neighbors=k_value)
        knn.fit(X_train, y_train)
        
        # Dự đoán
        y_pred = knn.predict(X_test)
        
        # Tính các metrics
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
    
    # Hiển thị kết quả
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("Recall", f"{recall:.3f}")
    with col3:
        st.metric("Precision", f"{precision:.3f}")
    
    # Hiển thị kết quả dự đoán
    st.subheader("Prediction Results")
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    results_df['Correct'] = results_df['Actual'] == results_df['Predicted']
    st.dataframe(results_df)