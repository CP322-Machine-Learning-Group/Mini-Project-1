# Imports
import numpy as np

# Class Object
class LogReg:
    def __init__(self, learning_rate=0.01, num_epochs=1000):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.bias = 0.0
        self.theta = None

    # The sigmoid Function
    def sigmoid_function(self, z):
        return 1 / (1 + np.exp(-z))

    # Parameters initialization
    def initialize_parameters(self, n_features):
        self.theta = np.zeros(n_features)

    # Accuracy Function
    def evaluate_acc(self, X, y, threshold=0.5):
        accuracy = self.predict(X, threshold)
        return np.mean(accuracy == y)

    # Fit Function
    def fit(self, X, y):
        m, n_features = X.shape
        self.initialize_parameters(n_features)

        loss_array = []

        for i in range(self.num_epochs):
            z = (X @ self.theta) + self.bias
            h = self.sigmoid_function(z)
            loss = self.logistic_loss(h, y)
            print("Epoch: {}, Loss: {}".format(i, loss))
            loss_array.append(loss)

            self.theta -= self.learning_rate * (1 / m) * np.sum(np.dot((h - y), X))

            # Update bias
            self.bias -= self.learning_rate * (1 / m) * np.sum(h - y)

        return loss_array

    # Predict and Test Function
    def predict(self, X, threshold=0.5):
        z = np.dot(X, self.theta) + self.bias
        h = self.sigmoid_function(z)
        return (h >= threshold).astype(int)

    '''
    def mean_squared_error(self, X, y):
        m = X.shape[0]
        z = np.dot(X, self.theta) + self.bias
        h = self.sigmoid_function(z)
        mse = np.mean((h - y) ** 2)
        return mse
    '''

    def logistic_loss(self, h, y):
        m = len(y)
        epsilon = 1e-15  # Small value to avoid division by zero in log
        loss = -(1 / m) * (np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon)))
        return loss
