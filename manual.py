import numpy as np
import math
import pandas as pd




def sigmoid(z):
    z = np.array(z)
    return 1/(1 + math.e**(-z))   # np.exp raises every element

def initialize_params(m_features, seed=None):   
    if seed is not None:
        np.random.seed(seed)
    w = np.random.randn(m_features, 1) * 0.01  # Extremely small initialized numbers; m by 1 vector
    b = 0
    return w, b

# Returns z and y_hat
def forward(X, w, b):   # CHECK IF THIS SHIT WORKS CORRECTLY
    z = X @ w + b   # @ does matrix multiplication
    z = np.array(z)
    y_hat = sigmoid(z)
    return z, y_hat

# Computes binary cross-entropy cost
def compute_cost(y, y_hat):
    n = len(y)
    cost = -(1/n) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return cost

def compute_gradients(X, y, y_hat):
    n = len(y)
    dw = (1/n) * X.T @ (y_hat - y)    # (m,1)
    db = (1/n) * np.sum(y_hat - y)    # scalar
    return dw, db

def update_params(w, b, dw, db, lr):    # lr stands for learning rate
    w = w - lr * dw
    b = b - lr * db
    return w, b

def train(X, y, lr, n_epochs, X_val=None, y_val=None):
    m = len(X[0])
    w, b = initialize_params(m, 123)
    trainingCosts = []
    validationCosts = []
    for epoch in range(n_epochs):
        z, y_hat = forward(X, w, b)
        cost = compute_cost(y, y_hat)
        trainingCosts.append(cost)
        dw, db = compute_gradients(X, y, y_hat)
        w, b = update_params(w, b, dw, db, lr)
    return w, b, trainingCosts

def predict_proba(X,w,b):
    z, y_hat = forward(X, w, b)
    return y_hat




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# P R E P R O C E S S I N G
# 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Grabbing training and validation data
train_data = pd.read_csv("./kaggle/input/titanic/train.csv")
test_data  = pd.read_csv("./kaggle/input/titanic/test.csv")

# Separating our target from our features
X = train_data.drop("Survived", axis=1)
y = train_data["Survived"]

# Removal of features unwanted for the model to use
X = X.drop("Ticket", axis=1)
X = X.drop("Embarked", axis=1)
X = X.drop("Cabin", axis=1)
X = X.drop("Fare", axis=1)
X = X.drop("Name", axis=1)
X = X.drop("PassengerId", axis=1)


# Normalizing all non-classified features
normalized = X.drop("Sex", axis=1)
normalized = normalized.fillna(normalized.mean()) # Replacing NaN values with their averages   
normalized = (normalized-normalized.mean())/normalized.std()

# One hot encoding
encoded = X["Sex"]
encoded = encoded.fillna(encoded.mode()) 
encoded = pd.get_dummies(encoded, drop_first=True)

# Combining the two
X = pd.concat([normalized, encoded], axis=1)

print(X.head())
# print(y.head())



X = X.to_numpy(dtype=float)
y = y.to_numpy(dtype=float).reshape(-1, 1) # (makes sure this is an n by 1 array, not 1D)
w, b, trainingCosts = train(X, y, 1, 100)
print(predict_proba(X[4], w, b))