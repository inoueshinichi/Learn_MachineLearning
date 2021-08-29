import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
    ADALINE, （バッチ）勾配降下法による学習規則、線形活性化関数を使用
'''

class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            # 線形活性化関数
            output = self.activation(X)
            # 誤差
            errors = (y - output)
            # 重みの更新
            update = self.eta * np.dot(X.T, errors)
            self.w_[1:] += update
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

if __name__ == '__main__':
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    #print(df.tail())

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    #plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='seota')
    #plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    #plt.xlabel('sepal length [cm]')
    #plt.ylabel('petal length [cm]')
    #plt.legend(loc='upper left')
    #plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    # 勾配降下法によるADALINEの学習 学習率=0.01
    ada1 = AdalineGD(n_iter=10, eta=0.01)
    ada1.fit(X, y)
    # エポック数とコストの関係を表す折れ線グラフのプロット
    ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error')
    ax[0].set_title('Adaline - Learning rate 0.01')

    # 勾配降下法によるADALINEの学習 学習率=0.0001
    ada2 = AdalineGD(n_iter=10, eta=0.0001)
    ada2.fit(X, y)
    # エポック数とコストの関係を表す折れ線グラフのプロット
    ax[1].plot(range(1, len(ada2.cost_)+1), np.log10(ada2.cost_), marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    plt.show()