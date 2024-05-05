import numpy as np

# Author: Daniel Vera morales
useArgmax = True


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(X, w):
    return sigmoid(np.matmul(X, w))


def gradient(X, Y, w):
    error = forward(X, w) - Y
    return np.matmul(X.T, error) / X.shape[0]


def loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.sum(first_term + second_term) / X.shape[0]


def classify(X, w):
    labels = forward(X, w)
    print(f"Using useArgmax={useArgmax}")
    if useArgmax:
        labels = np.argmax(labels, axis=1)
    return labels.reshape(-1, 1)


def binary_classify(X, w):
    return np.round(forward(X, w))
