import numpy as np

from lib.digit_classifier import classify, loss, gradient, binary_classify

# Author: Daniel Vera morales

is_binary_classification = False
apply_ravel = False


class Classifier:

    def __init__(self, encodeHelper):
        self.encodeHelper = encodeHelper
        print("Starting Digit Classifier...")

    def report(self, iteration, X_train, Y_train, X_test, Y_test, w):
        classified = self.apply_classification(X_test, w)
        matches = np.count_nonzero(classified == Y_test)
        n_test_examples = Y_test.shape[0]
        matches = matches * 100.0 / n_test_examples
        training_loss = loss(X_train, Y_train, w)
        print(f"{iteration} - Loss: {training_loss} - examples: {n_test_examples}, \n {matches}%")
        return training_loss

    def apply_classification(self, X_test, w):
        if is_binary_classification:
            classified = classify(X_test, w)
        else:
            classified = binary_classify(X_test, w)
        classified = self.encodeHelper.decode_value(classified)
        if apply_ravel:
            classified = np.ravel(classified)
        return classified

    def train(self, X_train, Y_train, X_test, Y_test, iterations, lr):
        w = np.zeros((X_train.shape[1], Y_train.shape[1]), dtype=np.float64)
        historyW = []
        for i in range(iterations):
            training_loss = self.report(i, X_train, Y_train, X_test, Y_test, w)
            w -= gradient(X_train, Y_train, w) * lr
            historyW.append([w, training_loss])
        self.report(iterations, X_train, Y_train, X_test, Y_test, w)
        return w, historyW

    def test(self, X, Y, w):
        total_examples = X.shape[0]
        correct_results = np.sum(self.apply_classification(X, w) == Y)
        success_pct = correct_results * 100 / total_examples
        print(f"Success: {correct_results}/{total_examples} {success_pct}%")
        return success_pct
