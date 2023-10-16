# -*- coding: utf-8 -*-
"""LG-HW.ipynb

Author: @Innocent Boakye Ababio

Colab file is located at
    https://colab.research.google.com/drive/1mEDUJDHOhTq3TD0x9wTTz1vIvXT9Q-Fu
"""

# ==================================
# Logistic Regression from scratch
# ==================================


# Import libraries
import numpy as np
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, recall_score
from pathlib import Path

#  ================================
# Downloading the "Pinma Indians Diabetes" data set
# ================================

# Create a directory if it doesn't exist
data_dir = Path('data')
data_dir.mkdir(parents=True, exist_ok=True)

# URL of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'

# Path to save the downloaded dataset
file_path = data_dir / 'pima-indians-diabetes.csv'

# Download the dataset if it doesn't exist
if not file_path.is_file():
    response = requests.get(url)
    with open(file_path, 'wb') as file:
        file.write(response.content)



#  ================================
# Process data set
# ================================
df = pd.read_csv('data/pima-indians-diabetes.csv')
X = df.iloc[:, :-1].values
# Normalize X inputs data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
# assign labels to y
y = df.iloc[:, -1].values



#  ================================
# Define sigmoid  and the Logistic Regression class
# ================================

def sigmoid(x):
    # Sigmoid activation function
    return 1 / (1 + np.exp(-x))

# logistic regression class
class LogisticRegression():

    def __init__(self, lr=1e-4, n_iters=10000):
        # Constructor method to initialize logistic regression model
        self.lr = lr  # Learning rate for gradient descent
        self.n_iters = n_iters  # Number of iterations for gradient descent
        self.weights = None  # Coefficients for features
        self.bias = None  # Intercept term

    def fit(self, X, y):
        # Method to train the logistic regression model
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights with zeros
        self.bias = 0  # Initialize bias with zero

        for _ in range(self.n_iters):
            # Compute linear predictions and apply sigmoid activation
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            # Compute gradients with respect to weights and bias
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)

            # Update weights and bias using gradient descent
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        # Method to make predictions using the trained model
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)

        # Convert probabilities to binary predictions (0 or 1)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred


# =====================
# Train the model
# =====================

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize logistic regression classifier with a specific learning rate
clf = LogisticRegression(lr=(1e-4)+0.9981414597976055)

# Train the logistic regression model on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Define a function to calculate accuracy
def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)

# Calculate and print the accuracy of the model on the test data
acc = accuracy(y_pred, y_test)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)

# Calculate recall
recall = recall_score(y_test, y_pred)

print("================================")
print("Accuracy:", acc)
print("F1 Score:", f1)
print("Recall:", recall)
print("================================")