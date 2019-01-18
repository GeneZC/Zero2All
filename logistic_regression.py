# -*- coding: utf-8 -*-

import numpy as np
import math

'''
Logistic Regression Algorithm
'''
class LogisticRegression():
    def __init__(self, learning_rate=.1):
        self.w = None
        self.b = None
        self.learning_rate = learning_rate

    def _init_params(self, X):
        n_features = np.shape(X)[1]
        # 初始化参数范围 [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features,))
        self.b = np.zeros((1,))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, X, y, n_iterations=1000):
        self._init_paras(X)
        for i in range(n_iterations):
            hypo = X.dot(self.w + self.b) #+ self.b
            y_pred = self.sigmoid(hypo)
            # loss
            # Note: the true loss should be 
            # err = -y*np.log(y_pred)-(1-y)*np.log(1-y_pred)
            # here just for simplification to calculate gradient
            err = y_pred - y
            self.w -= self.learning_rate * err.dot(X) / X.shape[0]
            self.b -= self.learning_rate * err.sum() / X.shape[0]

    def predict(self, X):
        y_pred = np.round(self.sigmoid(X.dot(self.w) + self.b))
        return y_pred.astype(int)


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    clf = LogisticRegression()
    clf.train(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)
