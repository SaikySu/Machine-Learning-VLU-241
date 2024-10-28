# Bài thực hành lab6
# BT1
## 1. Công nghệ sử dụng
Sử dụng thư viện: Streamlit, PyTorch, Matplotlib, NumPy.


## 2. Thuật toán sử dụng
Loss Functions: Mean Square Error (MSE)
Công thức: MSE = (1/n) * Σ(y_pred - y_true)²
Đặc điểm: Đo lường khoảng cách trung bình bình phương giữa giá trị dự đoán và giá trị thực


Binary Cross Entropy:
Công thức: BCE = -(1/N) * Σ(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))
Đặc điểm: Đo lường loss cho bài toán phân loại nhị phân
Use case: Binary classification problems

## 3. Kết quả:
<p align="center">
    <img src="https://github.com/SaikySu/Machine-Learning-VLU-241/blob/main/Lab%206/img/image.png">
</p>

# BT2
## 1. Công nghệ sử dụng
Sử dụng thư viện: Streamlit, PyTorch, Matplotlib, NumPy.


## 2. Thuật toán sử dụng
Activation Functions:
Đặc điểm:
Output range: (0,1)
Thường dùng cho lớp output trong binary classification

ReLU (Rectified Linear Unit):
Công thức: f(x) = max(0,x)
Đặc điểm:
Đơn giản và hiệu quả
Giải quyết vấn đề vanishing gradient

Tanh (Hyperbolic Tangent)
Công thức: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
Đặc điểm:
Output range: (-1,1)
Zero-centered

Softmax:
Công thức: softmax(x_i) = e^(x_i)/Σ(e^(x_j))
Đặc điểm:
Chuyển đổi vector input thành vector probability
Tổng các output = 1
Thường dùng cho lớp output trong multi-class classification

## 3. Kết quả:
<p align="center">
    <img src="https://github.com/SaikySu/Machine-Learning-VLU-241/blob/main/Lab%206/img/image2.png">
</p>

<p align="center">
    <img src="https://github.com/SaikySu/Machine-Learning-VLU-241/blob/main/Lab%206/img/image3.png">
</p>

<p align="center">
    <img src="https://github.com/SaikySu/Machine-Learning-VLU-241/blob/main/Lab%206/img/image4.png">
</p>

<p align="center">
    <img src="https://github.com/SaikySu/Machine-Learning-VLU-241/blob/main/Lab%206/img/image5.png">
</p>

