# -*- coding: utf-8 -*-

import numpy as np
import random

'''
Perceptron Algorithm
'''
class Perceptron(object):
    def __init__(self, learning_rate=0.1, update_schema='bgd', is_dual=True):
        '''
        param learning_rate: float, learning rate of updating, default 0.001
        param update_schema: str, schema of updating algorithm, in ['sgd', 'bgd'], default 'sgd'
        param is_dual: bool, use dual form of perceptron or not, default True
        return None
        '''
        self.lr = learning_rate
        self.update_schema = update_schema
        self.is_dual = is_dual
        self.W = None
        self.a = None
        self.b = None

    def train(self, X, Y, iteration=1000):
        '''
        param X: numpy.array, [None, size of feature vector], batches of feature vectors
        params Y: numpy.array, [None, 1], batches of labels
        return trained parameters
        '''
        data_num = X.shape[0]
        feature_size = X.shape[1]
        it = 0

        if self.is_dual:
            gram_matrix = np.matmul(X, np.transpose(X))
            self.a = np.zeros(data_num)
            self.b = np.zeros(1)
            while it <= iteration:
                it += 1
                temp = np.matmul(gram_matrix, self.a*Y)+self.b
                wrong = []
                for i, (x, y) in enumerate(zip(X, Y)):
                    if y*temp[i] <= 0:
                        self.a[i] += self.lr
                        self.b += self.lr*y
                        wrong.append((x, y))
                if len(wrong) == 0:
                    break
                else:
                    self.W = np.matmul(X.T, self.a*Y)
                    print("Iteration: "+str(it)+" W: "+str(self.W)+" b: "+str(self.b))
            return self.W, self.b
        else:
            if self.update_schema == 'sgd':
                self.W = np.random.randn(feature_size) / np.sqrt(0.5)
                self.b = np.zeros(1)
                while it <= iteration:
                    it += 1
                    wrong = []
                    for x, y in zip(X, Y):
                        if y*(np.dot(self.W, x)+self.b) <= 0:
                            wrong.append((x, y))
                    if len(wrong) == 0:
                        break
                    else:
                        _choice = random.choice(wrong)
                        self.W += self.lr*_choice[1]*_choice[0]
                        self.b += self.lr*_choice[1]
                        print("Iteration: "+str(it)+" W: "+str(self.W)+" b: "+str(self.b))
            elif self.update_schema == 'bgd':
                self.W = np.random.randn(feature_size) / np.sqrt(0.5)
                self.b = np.zeros(1)
                while it <= iteration:
                    it += 1
                    wrong = []
                    for x, y in zip(X, Y):
                        if y*(np.dot(self.W, x)+self.b) <= 0:
                            wrong.append((x, y))
                    if len(wrong) == 0:
                        break
                    else:
                        update_X = np.array([x[0] for x in wrong])
                        update_Y = np.array([x[1] for x in wrong])
                        self.W += self.lr*np.mean(update_X*update_Y, axis=0)
                        self.b += self.lr*np.mean(update_Y, axis=0)
                        print("Iteration: "+str(it)+" W: "+str(self.W)+" b: "+str(self.b))
            else:
                print("Error unknown update schema.")
            return self.W, self.b

    def predict(self, X):
        Y = np.sign(np.matmul(X, self.W) + self.b)
        return Y

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = -1
    y[y == 2] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    clf = Perceptron(update_schema='sgd', is_dual=False)
    clf.train(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)