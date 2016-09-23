# Dependencies
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

# Implementation


class LinearRegression(BaseEstimator, RegressorMixin):

    def __init__(self, eta=0.1, n_iter=100, C=0, method='batch'):
        self.eta_ = eta
        self.n_iter_ = n_iter
        self.C_ = C
        self.method_ = method

    def _net_input(self, X):
        return X @ self.w_[1:] + self.w_[0]

    def _loss_function(self, X, y):
        return 1 / (2 * y.size) * np.power(self._net_input(X) - y, 2).sum()

    def fit(self, X, y):
        self.errors_ = []
        self.w_ = np.zeros(X.shape[1] + 1)

        if self.method_ == 'batch':
            self._batch_gradient(X, y)

        if self.method_ == 'stoch':
            self._stoch_gradient(X, y)

        if self.method_ == 'mini-batch':
            pass

    def _batch_gradient(self, X, y):
        for i in range(self.n_iter_):
            self._step(X, y)

    def _step(self, X, y):
        error = self._net_input(X) - y
        self.w_[0] = self.w_[0] - self.eta_ / y.size * error.sum()
        if self.method_ == 'batch':
            self.w_[1:] = self.w_[1:] - self.eta_ / y.size * X.T @ error
        elif self.method_ == 'stoch':
            self.w_[1:] = self.w_[1:] - self.eta_ / y.size * X * error

        self.errors_.append(self._loss_function(X, y))

    def _stoch_gradient(self, X, y):
        for i in range(self.n_iter_):
            rand_indices = np.random.permutation(y.size)
            rand_row = np.random.randint(0, y.size)
            sample = X[rand_indices, :][rand_row, :]
            y_sample = y[rand_indices][rand_row]

            self._step(sample, y_sample)

    def predict(self, X):
        return self._net_input(X)
