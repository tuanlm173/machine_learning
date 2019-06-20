import numpy as np


class GDLinearRegression:
    """
    """

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.__learning_rate = learning_rate
        self.__iterations = iterations

    def fit(self, x_train, y_train):
        """Main function for Gradient Descent"""
        self.__x_train = x_train
        self.__y_train = y_train
        self.__costs = []
        self.__thetas = []
        self.__theta = np.zeros([self.__x_train.shape[1], 1])
        for i in range(self.__iterations):
            self.__theta = self.__theta - (self.__learning_rate * (1 / len(self.__x_train)) * np.dot(self.__x_train.T, (np.dot(self.__x_train, self.__theta) - self.__y_train)))
            self.__thetas.append(self.__theta)
            prediction = np.dot(self.__x_train, self.__theta)
            error = prediction - self.__y_train
            cost = 1 / (2 * len(self.__x_train)) * np.dot(error.T, error)
            self.__costs.append(cost)
            if i > 0 and (self.__costs[i - 1] - self.__costs[i]) < 1e-10: # early stop
                break
        return self

    def predict(self, x_test):
        """Predict values of independent variable"""
        self.__x_test = x_test
        return np.dot(self.__x_test, self.__theta) 

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self.__learning_rate = value

    @property
    def iterations(self):
        return self.__iterations

    @iterations.setter
    def iterations(self, value):
        self.__iterations = value

    @property
    def theta(self):
        return [self.__thetas[index].T for index in range(len(self.__thetas))]

    @property
    def costs(self):
        return [self.__costs[index][0][0] for index in range(len(self.__costs))]
