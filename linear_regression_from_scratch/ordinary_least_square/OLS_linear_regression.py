import numpy as np
import sklearn.datasets

"""
Example:
|----------------------------------|
import matplotlib.pyplot as plt
import OLSLinearRegression as olr

x, y = olr.sklearn.datasets.make_regression(n_samples=200, n_features=1, noise=20)

ols_lr_builder = olr.OLSLinearRegression()
reg = ols_lr_builder.fit(x,y)
y_pred = reg.predict(x)

plt.scatter(x,y)
plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color='red')
plt.show()
|----------------------------------|
"""


class OLSLinearRegression:
    """
    """

    def __repr__(self):
        return "OLS Linear Regression instance"

    def fit(self, x_train, y_train):
        """Fit linear model with OLS method"""
        self.__x_train = x_train
        self.__y_train = y_train

        x_mean = np.mean(self.__x_train)
        y_mean = np.mean(self.__y_train)

        numerator = 0
        denominator = 0

        for i in range(len(self.__x_train)):
            numerator += (self.__x_train[i] - x_mean) * (self.__y_train[i] - y_mean)
            denominator += np.square(self.__x_train[i] - x_mean)

        self.__slope = numerator / denominator
        self.__intercept = y_mean - self.__slope * x_mean
        return self

    def predict(self, x_test):
        """Predict using OLS linear model"""
        self.__x_test = x_test
        self.__predict = self.__slope * self.__x_test + self.__intercept
        return self.__predict

    @property
    def slope(self):
        return self.__slope

    @property
    def intercept(self):
        return self.__intercept
