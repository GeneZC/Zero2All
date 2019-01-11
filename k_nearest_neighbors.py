# -*- coding: utf-8 -*-

import numpy as np
from collections import namedtuple
from pprint import pformat
from sklearn.metrics.pairwise import euclidean_distances

class KNN(object):
    def __init__(self, k=5):
        self.k = k

    def train(self, X, Y):
        pass
        
    def predict(self, X):
        pass

class VallinaKNN(KNN):
    def train(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, X):
        Y = np.empty(X.shape[0])
        distances = euclidean_distances(X, self.X_train)
        distances_argsort = distances.argsort(axis=1)
        k_min_indices = np.argwhere(distances_argsort<self.k)[:,1].reshape((-1,self.k))
        labels = self.Y_train[k_min_indices]
        for i in range(len(X)):
            Y[i] = np.bincount(labels[i]).argmax()
        return Y

class Node(namedtuple('Node', 'location label left_child right_child')):
    def __repr__(self):
        return pformat(tuple(self))

'''
A simplified version, which does not check whether
the circle formalized by the testing point and current 
nearest point is intersecting with parent border or not.
'''
class KDTree(KNN):
    def _distance(self, x, y):
        return np.linalg.norm(x-y, ord=2)

    @staticmethod
    def _fit(X, Y, depth=0):
        try:
            k = X.shape[1]
        except IndexError as e:
            return None
        
        axis = depth % k
        X = X[X[:, axis].argsort()]
        median = X.shape[0] // 2

        try:
            X[median]
        except IndexError:
            return None

        return Node(
            location=X[median],
            label=Y[median],
            left_child=KDTree._fit(X[:median], Y, depth + 1),
            right_child=KDTree._fit(X[median + 1:], Y, depth + 1)
        )

    def _search(self, point, tree=None, depth=0, best=None):
        if tree is None:
            return best

        k = point.shape[0]
        # update best
        if best is None or self._distance(point, tree.location) < self._distance(best, tree.location):
            next_best = tree.label
        else:
            next_best = best

        # update branch
        if point[depth%k] < tree.location[depth%k]:
            next_branch = tree.left_child
        else:
            next_branch = tree.right_child
        return self._search(point, tree=next_branch, depth=depth+1, best=next_best)

    def train(self, X, Y):
        self.root = KDTree._fit(X, Y)

    def predict(self, X):
        Y = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            Y[i] = self._search(X[i], self.root)
        return Y


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    data = datasets.load_iris()
    X = normalize(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf = KDTree(k=4)
    clf.train(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)
