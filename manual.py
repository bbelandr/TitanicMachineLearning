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
def forward(X, w, b):
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
    for epoch in range(n_epochs):   # Training the model
        z, y_hat = forward(X, w, b)
        # print(type(y_hat[0][0]))
        cost = compute_cost(y, y_hat)
        trainingCosts.append(cost)
        dw, db = compute_gradients(X, y, y_hat)
        w, b = update_params(w, b, dw, db, lr)
    y_hat = predict_proba(X_val, w, b)
    # print(type(y_hat[0][0]))
    cost = compute_cost(y_val, y_hat)
    validationCosts.append(cost)
    return w, b, trainingCosts, validationCosts

def predict_proba(X,w,b):
    z, y_hat = forward(X, w, b)
    return y_hat




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# P R E P R O C E S S I N G
# 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def preprocess (dataset, toRemove, toNormalize, toEncode):
    dataset = dataset.drop(toRemove, axis=1) # Removing unwanted features

    # Normalization
    normalized = dataset.drop(toEncode, axis=1)
    normalized = normalized.fillna(normalized.mean()) # Replacing NaN values with their averages   
    normalized = (normalized-normalized.mean())/normalized.std()

    # Encoding
    encoded = dataset.drop(toNormalize, axis=1)
    encoded = encoded.fillna(encoded.mode()) 
    encoded = pd.get_dummies(encoded, drop_first=True)

    fullSet = pd.concat([normalized, encoded], axis=1)
    return fullSet

# Grabbing training and validation data
train_data = pd.read_csv("./kaggle/input/titanic/train.csv")
test_data  = pd.read_csv("./kaggle/input/titanic/test.csv")



# Separating our target from our features
X_train = train_data.drop("Survived", axis=1)
y_train = train_data["Survived"]

# Shuffle row indices
np.random.seed(42)  # reproducibility
indices = np.arange(len(train_data))
np.random.shuffle(indices)

# 80/20 split index
split = int(0.8 * len(indices))

# Training and validation indices
train_idx, val_idx = indices[:split], indices[split:]

# Create train/val sets
X_train, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
y_train, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

X_train = preprocess(X_train, ["Ticket", "Embarked", "Cabin", "Fare", "Name", "PassengerId"], ["Pclass", "Age", "SibSp", "Parch"], "Sex")
X_val = preprocess(X_val, ["Ticket", "Embarked", "Cabin", "Fare", "Name", "PassengerId"], ["Pclass", "Age", "SibSp", "Parch"], "Sex")


# print(y_val)
# print(y.head())



X_train = X_train.to_numpy(dtype=float)
y_train = y_train.to_numpy(dtype=float).reshape(-1, 1) # (makes sure this is an n by 1 array, not 1D)
X_val = X_val.to_numpy(dtype=float)
y_val = y_val.to_numpy(dtype=float).reshape(-1, 1) # (makes sure this is an n by 1 array, not 1D)
w, b, trainingCosts, validationCosts = train(X_train, y_train, 1, 100, X_val, y_val)
# print(trainingCosts, validationCosts)

X_test = preprocess(test_data, ["Ticket", "Embarked", "Cabin", "Fare", "Name", "PassengerId"], ["Pclass", "Age", "SibSp", "Parch"], "Sex")
y_prediction = predict_proba(X_train, w, b)
y_prediction = pd.Series(y_prediction.ravel())
for i in range(len(y_prediction)):
    if y_prediction[i] > 0.5:
        y_prediction[i] = 1
    else:
        y_prediction[i] = 0
print(test_data)
output = pd.DataFrame({
    "PassengerId": test_data.PassengerId,
    "Survived": pd.Series(y_prediction.ravel())
})
output[:418].to_csv("submission.csv", index=False)
# This submission technically isn't even finished; I had to go back through the submission.csv file and remove column 0's decimals so that it would be accepted