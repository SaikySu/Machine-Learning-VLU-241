import streamlit as st
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt

def calculate_svm(x, y, C):
    N = x.shape[0]
    
    # Construct the matrices required for QP in standard form
    H = np.dot(y * x, (y * x).T)
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(np.ones(N) * -1)
    A = cvxopt_matrix(y.reshape(1, -1).astype(np.double))
    b = cvxopt_matrix(np.zeros(1))
    
    g = np.vstack([-np.eye(N), np.eye(N)])
    G = cvxopt_matrix(g)
    
    h1 = np.hstack([np.zeros(N), np.ones(N) * C])
    h = cvxopt_matrix(h1)
    
    # solver parameters
    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10
    
    # Perform QP
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    
    # the solution to the QP, λ
    lamb = np.array(sol['x'])
    
    # Calculate w using the lambda, which is the solution to QP
    w = np.sum(lamb * y * x, axis=0)
    
    # Find support vectors
    sv_idx = np.where(lamb > 1e-5)[0]
    sv_x = x[sv_idx]
    sv_y = y[sv_idx]
    
    # Calculate b using the support vectors and calculate the average
    b = np.mean(sv_y - np.dot(sv_x, w))
    
    return w, b, lamb, sv_x, sv_y

def plot_svm(x, y, w, b, lamb, sv_x, C):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    color = ['red' if a == 1 else 'blue' for a in y]
    ax.scatter(x[:, 0], x[:, 1], s=200, c=color, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Visualize the decision boundary
    x1_dec = np.linspace(0, 1, 100)
    x2_dec = -(w[0] * x1_dec + b) / w[1]
    ax.plot(x1_dec, x2_dec, c='black', lw=1.0, label='decision boundary')
    
    # Display slack variables
    y_hat = np.dot(x, w) + b
    slack = np.maximum(0, 1 - y_hat * y.flatten())
    
    for s, (x1, x2) in zip(slack, x):
        ax.annotate(f"{s:.2f}", (x1, x2), xytext=(5, 5), textcoords='offset points')
    
    # Visualize the positive & negative boundary
    w_norm = np.linalg.norm(w)
    half_margin = 1 / w_norm
    
    upper = x2_dec + half_margin
    lower = x2_dec - half_margin
    
    ax.plot(x1_dec, upper, '--', lw=1.0, label='positive boundary')
    ax.plot(x1_dec, lower, '--', lw=1.0, label='negative boundary')
    
    ax.scatter(sv_x[:, 0], sv_x[:, 1], s=60, marker='o', facecolors='none', edgecolors='g')
    ax.legend()
    ax.set_title(f'C = {C:.1f}, Σξ = {np.sum(slack):.2f}')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    
    return fig

# Streamlit app
st.title('Support Vector Machine (SVM) Visualization')

# Data
x = np.array([[0.2, 0.869], [0.687, 0.212], [0.822, 0.411], [0.738, 0.694], [0.176, 0.458],
              [0.306, 0.753], [0.936, 0.413], [0.215, 0.410], [0.612, 0.375], [0.784, 0.602],
              [0.612, 0.554], [0.357, 0.254], [0.204, 0.775], [0.512, 0.745], [0.498, 0.287],
              [0.251, 0.557], [0.502, 0.523], [0.119, 0.687], [0.495, 0.924], [0.612, 0.851]])

y = np.array([-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1])
y = y.reshape(-1, 1)

# Sidebar for parameter tuning
C = st.sidebar.slider('C (regularization parameter)', 0.1, 100.0, 50.0, 0.1)

# Calculate SVM
w, b, lamb, sv_x, sv_y = calculate_svm(x, y, C)

# Plot results
fig = plot_svm(x, y, w, b, lamb, sv_x, C)
st.pyplot(fig)

# Display additional information
st.write(f"Number of support vectors: {len(sv_x)}")
st.write(f"w = {w}")
st.write(f"b = {b:.4f}")