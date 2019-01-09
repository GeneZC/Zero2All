# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt

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
                temp = np.matmul(gram_matrix, self.a*Y.squeeze(-1))+self.b
                wrong = []
                for i, (x, y) in enumerate(zip(X, Y)):
                    if y*temp[i] <= 0:
                        self.a[i] += self.lr
                        self.b += self.lr*y
                        wrong.append((x, y))
                if len(wrong) == 0:
                    break
                else:
                    self.W = np.matmul(X.T, self.a*Y.squeeze(-1))
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


            

def generate_data(w, b, data_num):
    w = np.array(w)
    feature_size = len(w)
    X = np.random.rand(data_num, feature_size) * 20  #随机产生numlines个数据的数据集
    Y = np.sign(np.sum(w*X,axis=1)+b)    #用标准线 w*x+b=0进行

    #下面是存储标准分类线，以便显示观察
    x = np.linspace(0, 20, 500)      #创建分类线上的点，以点构线。
    y = -w[0] / w[1] * x - b / w[1]
    rows = np.column_stack((x.T, y.T))
    X = np.row_stack((X, rows))
    Y = np.row_stack((np.expand_dims(Y, axis=1), np.zeros((500, 1))))

    return X, Y

def show_figure(X, Y):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('Linear separable data set')
    plt.xlabel('X')
    plt.ylabel('Y')
    #图例设置
    labels = ['classOne', 'standardLine', 'classTow', 'modelLine']
    markers = ['o','.','x','.']
    colors = ['r','y','g','b']
    for i in range(4):
        idx = np.where(Y[:]==i-1)   #找出同类型的点，返回索引值
        ax.scatter(X[idx, 0], X[idx, 1], marker=markers[i], color=colors[i], label=labels[i], s=10)

    plt.legend(loc = 'upper right')
    plt.show()

def test():
    X, Y = generate_data([1,-2], 7, 200)
    perceptron = Perceptron(is_dual=True)
    w, b = perceptron.train(X, Y)
    x = np.linspace(0,20,500)    #创建分类线上的点，以点构线。
    y = -w[0]/w[1]*x - b/w[1]
    rows = np.column_stack((x.T,y.T))
    X = np.row_stack((X,rows))
    Y = np.row_stack((Y, 2*np.ones((500,1))))
    show_figure(X, Y)

if __name__ == '__main__':
    test()
