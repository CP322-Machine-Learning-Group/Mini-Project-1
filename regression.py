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
    

class KNN:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

def evaluate_acc(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    return accuracy

def one_hot(X):
    encoded_features = np.zeros(X.shape)
    for i, column in enumerate(X.T):
        categories = np.unique(column)
        category_to_index = {category: idx for idx, category in enumerate(categories)}
        for j, val in enumerate(column):
            if val in category_to_index:
                encoded_features[j, i] = int(category_to_index[val])
    return encoded_features

def k_fold_cross_validation(model, X, y, k):
    kf = KFold(n_splits=k)
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = evaluate_acc(y_test, y_pred)
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    return mean_accuracy
