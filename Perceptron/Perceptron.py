import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from Plot_decision_regions import plot_decision_regions

print(np.__version__)
print(sys.version)

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter): 
            errors = 0
            for xi, target in zip(X, y):
                # 重みの更新 w1~wn
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                # 重みの更新 w0
                self.w_[0] += update
                # 重みの更新が0出ない場合は、対象サンプルを誤分類としてカウント
                errors += int(update != 0.0)
            # 反復回数毎の誤差を格納
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        '''
            [N(サンプル数)xM(特徴数)の特徴行列] ・ [M次元のパラメータベクトル] + [N次元のW0]
        '''
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


if __name__ == "__main__":
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    print(df.tail())

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='seota')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    print(ppn.errors_)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()

    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

