# Imports.
import numpy as np

# Class Object.
class LogReg:
    def __init__(self, learning_rate=0.01, num_epochs=1000): # Learning Rate and Epochs can be changed.
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
    # The sigmoid Function.
    def sigmoid_function(self, z):
        return 1/ (1 + np.exp(-z))

    # Parameters initialization.
    def initialize_parameters(self, n_features):
        self.theta = np.zeros(n_features)
        self.bias = 0

    # Accuracy Fuction.
    def evaluate_acc(self, x, y, threshold=0.5):
        accuracy = self.predict(x, threshold)
        return np.mean(accuracy == y)


    # Fit Function.
    def fit(self, x, y):
        m, n_features = x.shape
        self.initialize_parameters(n_features)

        for _ in range(self.num_epochs):
            z = np.dot(x, self.theta) + self.bias
            h = self.sigmoid(z)
            gradient = np.dot(x.T, (h - y)) / m
            self.theta -= self.learning_rate * gradient
            self.bias -= self.learning_rate * np.sum(h - y) / m

    # Predict and Test Function.
    def predict(self, x, threshold=0.5):
        z = np.dot(x, self.theta) + self.bias
        h = self.sigmoid(z)
        return (h >= threshold).astype(int)
