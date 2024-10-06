import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import string
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score

# Đọc file CSV được tải lên
file_path = 'E:\Learn\University Learn\Y3_HK1\Machine Learning\Practice\Code\Lab2\Education.csv'
data = pd.read_csv(file_path)

# Tiền xử lý dữ liệu văn bản
def preprocess_text(text):
    # Loại bỏ dấu câu và chuyển về chữ thường
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    return text

# Áp dụng tiền xử lý cho cột Text
data['Text'] = data['Text'].apply(preprocess_text)

# Chuyển đổi nhãn 'positive' và 'negative' sang dạng số
label_encoder = LabelEncoder()
data['Label'] = label_encoder.fit_transform(data['Label'])

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['Label'], test_size=0.2, random_state=42)

# Biểu diễn dữ liệu văn bản thành vector
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

X_train_vec.shape, X_test_vec.shape, y_train.shape, y_test.shape
# Phân phối Bernoulli
bernoulli_nb = BernoulliNB()
bernoulli_nb.fit(X_train_vec, y_train)
y_pred_bernoulli = bernoulli_nb.predict(X_test_vec)
accuracy_bernoulli = accuracy_score(y_test, y_pred_bernoulli)

# Phân phối Multinomial
multinomial_nb = MultinomialNB()
multinomial_nb.fit(X_train_vec, y_train)
y_pred_multinomial = multinomial_nb.predict(X_test_vec)
accuracy_multinomial = accuracy_score(y_test, y_pred_multinomial)

title = st.text_input("Tôi sẽ phân tích tâm trạng của bạn ", "0")

user = vectorizer.transform(np.array([title]))
ans = bernoulli_nb.predict(user)

st.write("Tâm trạng của bạn là", "tốt" if ans == "positive" else "không tốt")
