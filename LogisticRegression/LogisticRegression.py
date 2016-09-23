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
        self.n_iter_ = n_iter
        self.C_ = C
        self.method_ = method

    def fit(self, X, y):
        # self.errors_ = []
        self.all_w_ = np.zeros((np.unique(y).size, X.shape[1] + 1))
        # self.w_ = np.zeros(X.shape[1] + 1)
        self.num_labels = np.unique(y).size

        if self.method_ == 'batch':
            self._batch_grad(X, y)
        if self.method_ == 'stoch':
            self._stoch_grad(X, y)
        if self.method_ == 'mini-batch':
            self._minibatch_grad(X, y)
        return self

    def _minibatch_grad(self, X, y):
        pass

    def _minibatch_step(self, X, y):
        pass

    def _batch_grad(self, X, y):
        for i in range(self.n_iter_):
            self._batch_step(X, y)

    def _batch_step(self, X, y):
        self._one_v_rest(X, y)
        # self.errors_.append(self.logit_loss(X, y))

    def _one_v_rest(self, X, y):
        for c in np.unique(y):
            y_ovr = np.where(y == c, 1, 0)
            error = y_ovr - self.sigmoid(X, c)
            self.all_w_[c, 0] += self.eta_ * error.sum() - self.eta_ * \
                self.C_ * self.all_w_[c, 0]
            self.all_w_[c, 1:] += self.eta_ * X.T @ error - \
                self.eta_ * self.C_ * self.all_w_[c, 1:]

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

    def sigmoid(self, X, c):
        z = self.net_input(X, c)
        return (1 / (1 + np.exp(-z)))

    def net_input(self, X, c):
        return (X @ self.all_w_[c, 1:] + self.all_w_[c, 0])

    def logit_loss(self, X, y):
        return -(y * np.log(self.sigmoid(X)) + (1 - y) *
                 np.log(1 - self.sigmoid(X))).sum()

    def predict(self, X):
        probas = self._predict_one_v_rest(X)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X):
        return self._predict_one_v_rest(X)

    def _predict_one_v_rest(self, X):
        probas = np.zeros((X.shape[0], self.num_labels))
        for c in range(self.num_labels):
            probas[:, c] = self.sigmoid(X, c)
        return probas

    def score(self, y_pred, y_real):
        pass
        # return (1 - sum(list(map(lambda x, y: x != y, y_pred, y_real))) / y_real.shape[0])
