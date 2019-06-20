import numpy as np 


class GDLogisticRegression:
    """
    """

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.__learning_rate = learning_rate
        self.__iterations = iterations

    def __add_intercept(self, x):
        """Add intercept for Logistic Regression"""
        intercept = np.ones((x.shape[0], 1))
        return np.concatenate((intercept, x), axis=1)

    def __sigmoid(self, z):
        return 1 / ( 1 + np.exp(-z))

    def __loss(self, h, y):
        epsilon = 1e-5 
        return np.average(-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon))

    def fit(self, x, y):
        costs = []
        x = self.__add_intercept(x)
        self.__theta = np.zeros((x.shape[1], 1))
        z = np.dot(x, self.__theta)
        h = self.__sigmoid(z)
        for i in range(self.__iterations):
            gradient = np.dot(x.T, (h - y)) / y.size
            self.__theta = self.__theta - self.__learning_rate * gradient
            z = np.dot(x, self.__theta)
            h = self.__sigmoid(z)
            costs.append(self.__loss(h, y))
            if i > 0 and (costs[i - 1] - costs[i]) < 1e-15: # early stop
                break
        return costs, self.__theta
            
    def predict(self, x_test, threshold):
        self.__x_test = self.__add_intercept(x_test)
        return (self.__sigmoid(np.dot(self.__x_test, self.__theta)) >= threshold).astype(int)
        
