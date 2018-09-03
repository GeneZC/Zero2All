# -*- coding: utf-8 -*-

import numpy as np
import random

'''
Perceptron Algprithm
'''
class Perceptron(object):
    def __init__(self, batch_size=32, learning_rate=0.001, update_schema='sgd', is_dual=True):
        '''
        param batch_size: int, batch size when training, default 32
        param learning_rate: float, learning rate of updating, default 0.001
        param update_schema: str, schema of updating algorithm, in ['sgd', 'bgd'], default 'sgd'
        param is_dual: bool, use dual form of perceptron or not, default True
        return None
        '''
        self.batch_size = batch_size
        self.lr = learning_rate
        self.update_schema = update_schema
        self.is_dual = is_dual

    def fit(self, X, Y):
        '''
        param X: numpy.array, [None, size of feature vector], batches of feature vectors
        params Y: numpy.array, [None, 1], batches of labels
        return train_acc: float, training accuracy
        '''
        # TODO: initialize weights 
        feature_size = X.size(1)

        if self.update_schema == 'sgd':
            self.W = np.random.randn(feature_size) / np.sqrt(feature_size / 2)
            self.b = np.zeros((1))
            wrong = []
            while True:
                for x, y in zip(X, Y):
                    if y*(np.dot(self.W, x)+self.b) < 0:
                        wrong.append((x, y))
                if len(wrong) == 0:
                    break
                else:
                    _choice = random.choice(np.array(wrong))
                    self.W += self.lr*_choice[1]*_choice[0]
                    self.b += self.lr*_choice[1]

        elif self.update_schema == 'bgd':
            batch_num = int(X.size(0) / self.batch_size)
        else:
            print("Error unknown update schema.")



    def predict(self, X):
        pass
