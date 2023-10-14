# Imports.
import numpy as np

# Class Object.
class LogReg:
    def __init__(self, learning_rate=0.01, num_epochs=1000): # Learning Rate and Epochs can be changed.
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.theta = None
        self.bias = 0.0
        
    # The sigmoid Function.
    def sigmoid_function(self, z):
        return 1/ (1 + np.exp(-z))

    # Parameters initialization.
    def initialize_parameters(self, n_features):
        self.theta = np.zeros(n_features)
        self.bias = 0

    # Accuracy Fuction.
    def evaluate_acc(self, X, y, threshold=0.5):
        accuracy = self.predict(X, threshold)
        return np.mean(accuracy == y)

    # Fit Function.
    def fit(self, X, y):
        m, n_features = X.shape
        self.initialize_parameters(n_features)
        
        # Ensure y is a 1D array
        y = y.reshape(-1)
        
        for _ in range(self.num_epochs):
            z = np.dot(X, self.theta) + self.bias
            h = self.sigmoid_function(z)
            loss = self.logistic_loss(h, y)
            gradient = np.dot(X.T, (h - y)) / m
            self.theta -= self.learning_rate * gradient

            # Update bias
            self.bias -= self.learning_rate * np.sum(h - y) / m

    # Predict and Test Function.
    def predict(self, X, threshold=0.5):
        z = np.dot(X, self.theta) + self.bias
        h = self.sigmoid(z)
        return (h >= threshold).astype(int)
    
    def mean_squared_error(self, X, y):
        m = X.shape[0]
        z = np.dot(X, self.theta) + self.bias
        h = self.sigmoid_function(z)
        mse = np.mean((h - y) ** 2)
        return mse
    
    def logistic_loss(self, h, y):
        m = len(y)
        epsilon = 1e-15  # Small value to avoid division by zero in log
        loss = - (1 / m) * (np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon)))
        return loss


        
