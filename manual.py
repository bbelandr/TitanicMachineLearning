import numpy as np
import math
import pandas as pd

def sigmoid(z):
    return 1/(1 + math.e**(-z))

def initialize_params(m_features, seed=None):   
    if seed is not None:
        np.random.seed(seed)
    w = np.random.randn(m_features) * 0.01  # Extremely small initialized numbers; 1D array that we can treat as an m by 1 vector
    b = 0
    return w, b

# Returns z and y_hat
def forward(X, w, b):
    z = X @ w + b   # @ does matrix multiplication
    y_hat = sigmoid(z)
    return z, y_hat

# Computes binary cross-entropy cost
# def compute_cost(y, y_hat):

