import numpy as np


# Author: Daniel Vera morales

def prepend_bias(X):
    # ("axis=1" stands for: "insert a column, not a row")
    return np.insert(X, 0, 1, axis=1)


def extract_test_data(data_np, test_size=0.23):
    test_size = round(data_np.shape[0] * test_size)
    print(f"test_size: {test_size}")
    X_test = data_np[:test_size]
    X_train = data_np[test_size:]
    print(X_test)
    Y_train = X_train[:, -1]
    Y_test = X_test[:, -1]
    X_train = X_train[:, :-1]
    X_test = X_test[:, :-1]
    return X_train, Y_train, X_test, Y_test
