import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

def calculate_svm(x, y):
    # Calculate H matrix
    H = np.dot(y * x, (y * x).T)

    # Construct matrices for QP in standard form
    n = x.shape[0]
    P = matrix(H)
    q = matrix(-np.ones((n, 1)))
    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))

    # Set solver parameters
    solvers.options['show_progress'] = False
    solvers.options['abstol'] = 1e-10
    solvers.options['reltol'] = 1e-10
    solvers.options['feastol'] = 1e-10

    # Perform QP
    sol = solvers.qp(P, q, G, h, A, b)

    # Extract lambda (solution of QP)
    lamb = np.array(sol['x'])

    # Calculate w
    w = np.dot((y * x).T, lamb).flatten()

    # Calculate b (using average of support vectors)
    sv_idx = np.where(lamb > 1e-5)[0]
    b = np.mean(y[sv_idx] - np.dot(x[sv_idx], w))

    return lamb, w, b

def plot_svm(x, y, lamb, w, b):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)

    # Plot data points
    colors = ['red' if label == 1 else 'blue' for label in y.flatten()]
    ax.scatter(x[:, 0], x[:, 1], c=colors, s=200, zorder=5)

    # Plot decision boundary
    x1 = np.array([0, 4])
    x2 = (-w[0]*x1 - b) / w[1]
    ax.plot(x1, x2, 'k-', label='decision boundary')

    # Plot positive and negative boundaries
    margin = 1 / np.linalg.norm(w)
    x2_pos = (-w[0]*x1 - b + 1) / w[1]
    x2_neg = (-w[0]*x1 - b - 1) / w[1]
    ax.plot(x1, x2_pos, 'b--', label='positive boundary')
    ax.plot(x1, x2_neg, 'r--', label='negative boundary')

    # Annotate points with lambda values
    for i, (x_i, y_i) in enumerate(x):
        ax.annotate(f'λ={lamb[i][0]:.1f}', (x_i+0.1, y_i+0.1), fontsize=10)

    ax.legend()
    return fig

st.title('SVM Visualization')

# Input for data points
st.sidebar.header('Data Points')
x1 = st.sidebar.number_input('x1', value=1.0, step=0.1)
y1 = st.sidebar.number_input('y1', value=3.0, step=0.1)
x2 = st.sidebar.number_input('x2', value=2.0, step=0.1)
y2 = st.sidebar.number_input('y2', value=2.0, step=0.1)
x3 = st.sidebar.number_input('x3', value=1.0, step=0.1)
y3 = st.sidebar.number_input('y3', value=1.0, step=0.1)

# Define data points
x = np.array([[x1, y1], [x2, y2], [x3, y3]])
y = np.array([[1.], [1.], [-1.]])

# Calculate SVM
lamb, w, b = calculate_svm(x, y)

# Display results
st.write('lambda =', np.round(lamb.flatten(), 3))
st.write('w =', np.round(w, 3))
st.write('b =', np.round(b, 3))

# Plot SVM
fig = plot_svm(x, y, lamb, w, b)
st.pyplot(fig)

margin = 1 / np.linalg.norm(w)
st.write(f"Margin = {margin:.4f}")