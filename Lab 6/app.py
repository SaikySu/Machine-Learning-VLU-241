import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np

# Thiết lập trang
st.title("Các hàm Loss và Activation trong Deep Learning")

# Hàm Loss
def meanSquareError(output, target):
    return torch.mean((output - target) ** 2)

def binaryEntropyLoss(output, target, n):
    return torch.nn.functional.binary_cross_entropy(output, target, reduction='sum') / n

# Hàm Activation
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def relu(x):
    return torch.maximum(torch.tensor(0.0), x)

def tanh(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

def softmax(x):
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x)

# Chọn loại hàm
function_type = st.radio("Chọn loại hàm:", ["Loss Functions", "Activation Functions"])

if function_type == "Loss Functions":
    st.header("Hàm Loss")
    loss_type = st.selectbox("Chọn hàm loss:", ["Mean Square Error", "Binary Cross Entropy"])
    
    # Input data
    st.write("Nhập dữ liệu (cách nhau bởi dấu phẩy):")
    output_str = st.text_input("Output:", "0.1,0.3,0.6,0.7")
    target_str = st.text_input("Target:", "0.31,0.32,0.8,0.2")

    try:
        output = torch.tensor([float(x) for x in output_str.split(",")])
        target = torch.tensor([float(x) for x in target_str.split(",")])

        if loss_type == "Mean Square Error":
            loss = meanSquareError(output, target)
            st.write(f"Mean Square Error: {loss.item():.4f}")

        else:  # Binary Cross Entropy
            n = len(output)
            loss = binaryEntropyLoss(output, target, n)
            st.write(f"Binary Cross Entropy Loss: {loss.item():.4f}")

        # Vẽ biểu đồ
        fig, ax = plt.subplots()
        ax.plot(output.numpy(), label='Output')
        ax.plot(target.numpy(), label='Target')
        ax.legend()
        ax.set_title(loss_type)
        st.pyplot(fig)

    except:
        st.error("Vui lòng nhập dữ liệu hợp lệ (các số thực cách nhau bởi dấu phẩy)")

else:
    st.header("Hàm Activation")
    activation_type = st.selectbox("Chọn hàm activation:", 
                                 ["Sigmoid", "ReLU", "Tanh", "Softmax"])

    if activation_type != "Softmax":
        # Tạo dữ liệu cho biểu đồ
        x = torch.linspace(-5, 5, 100)
        
        if activation_type == "Sigmoid":
            y = sigmoid(x)
        elif activation_type == "ReLU":
            y = relu(x)
        else:  # Tanh
            y = tanh(x)

        # Vẽ biểu đồ
        fig, ax = plt.subplots()
        ax.plot(x.numpy(), y.numpy())
        ax.grid(True)
        ax.set_title(f"Hàm {activation_type}")
        st.pyplot(fig)

    else:  # Softmax
        # Input cho softmax
        input_str = st.text_input("Nhập giá trị đầu vào:", "1,2,3,4,5")
        try:
            x = torch.tensor([float(x) for x in input_str.split(",")])
            y = softmax(x)
            
            # Vẽ biểu đồ
            fig, ax = plt.subplots()
            ax.bar(range(len(y)), y.numpy())
            ax.set_title("Softmax Probabilities")
            ax.set_ylim(0, 1)
            st.pyplot(fig)
            
            # Hiển thị giá trị
            st.write("Probabilities:")
            for i, prob in enumerate(y):
                st.write(f"Index {i}: {prob.item():.4f}")
                
        except:
            st.error("Vui lòng nhập dữ liệu hợp lệ")