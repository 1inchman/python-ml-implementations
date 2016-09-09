#  Dependencies
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# Logistic regression class


class LogisticRegression(BaseEstimator, ClassifierMixin):

    def __init__(self, eta=0.1, C=0, n_iter=100, method='batch', verbose=False):
        # if method not in ['batch', 'mini-batch', 'stoch', 'scipy']:
        #     raise ValueError(
        #         'Method must be batch, mini-batch, stoch or scipy')
        self.eta_ = eta
        self.verbose = verbose
        self.n_iter_ = n_iter
        self.C_ = C
        self.method_ = method
        return self

    def fit(self, X, y):
        self.errors_ = []
        self.w_ = np.zeros(X.shape[1] + 1)

        if self.method_ == 'batch':
            self._batch_grad(X, y)
        if self.method_ == 'stoch':
            self._stoch_grad(X, y)
        if self.method_ == 'mini-batch':
            pass
        return self

    def _batch_grad(self, X, y):
        for i in range(self.n_iter_):
            self._batch_step(X, y)

    def _batch_step(self, X, y):
        error = y - self.sigmoid(X)
        self.w_[0] += self.eta_ * error.sum() - self.eta_ * \
            self.C_ * self.w_[0]
        self.w_[1:] += self.eta_ * X.T @ error - \
            self.eta_ * self.C_ * self.w_[1:]
        self.errors_.append(self.logit_loss(X, y))

    def _stoch_grad(self, X, y):
        for i in range(self.n_iter_):
            self._stoch_step(X, y)

    def _stoch_step(self, X, y):
        rand_indices = np.random.permutation(y.size)
        rand_choice = np.random.randint(0, y.size)
        sample = X[rand_indices, :][rand_choice, :]

        error = y[rand_indices][rand_choice] - self.sigmoid(sample)
        self.w_[0] += self.eta_ * error - self.eta_ * self.C_ * self.w_[0]
        self.w_[1:] += self.eta_ * sample * \
            error - self.eta_ * self.C_ * self.w_[1:]
        self.errors_.append(self.logit_loss(
            sample, y[rand_indices][rand_choice]))

    def sigmoid(self, X):
        z = self.net_input(X)
        return (1 / (1 + np.exp(-z)))

    def net_input(self, X):
        return (X @ self.w_[1:] + self.w_[0])

    def logit_loss(self, X, y):
        return -(y * np.log(self.sigmoid(X)) + (1 - y) *
                 np.log(1 - self.sigmoid(X))).sum()

    def predict(self, X):
        return np.where(self.sigmoid(X) > 0.5, 1, 0)

    def predict_proba(self, X):
        return self.sigmoid(X)

    def score(self, y_pred, y_real):
        return (1 - sum(list(map(lambda x, y: x != y, y_pred, y_real))) / y_real.shape[0])
