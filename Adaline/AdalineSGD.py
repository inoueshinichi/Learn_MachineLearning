from numpy.random import seed
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Perceptron import plot_decision_regions

class AdalineSGD(object):
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)
        self.cost_ = []
    
    def fit(self, X, y):
        """
            確率的勾配降下法によるモデルの学習 
        """
        self._initialize_weights(X.shape[1])  # 毎回初期化が行われる
        
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            # 各サンプルに対する計算
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """
            重みを再初期化することなくトレーニングデータに適合させる
        """
        if len(X.shape) > 1:
            X = X.reshape(X.shape[1]) # 型変換 (1,2) -> (2,)

        if not self.w_initialized:
            self._initialize_weights(X.shape[0])
                   
        cost = None
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                cost = self._update_weights(xi, y)
        else:
            cost = self._update_weights(X, y)
        self.cost_.append(cost)
        return self

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.activation(xi)
        error = target - output  # 各サンプルごとの誤差
        self.w_[1:] += self.eta * np.dot(xi, error)  # 学習係数 × 1サンプルの特徴量ベクトル × 誤差(スカラー) = ベクトル 
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        # 恒等関数
        return self.net_input(X)

    def predict(self, X):
        # 量子化器
        return np.where(self.activation(X) >= 0.0, 1, -1)


if __name__ == "__main__":
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    adaSGD = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    #adaSGD.fit(X_std, y)

    print(X_std.shape)
    # 再初期化なしの学習（オンライン向け）
    for _ in range(15):
        X_stdSh, ySh = adaSGD._shuffle(X_std, y)
        for xi, target in zip(X_stdSh, ySh):
            #print(xi.shape)
            adaSGD.partial_fit(xi, target)
    
    plot_decision_regions(X_std, y, classifier=adaSGD)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.plot(range(1, len(adaSGD.cost_) + 1), adaSGD.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost or Iter Cost')
    plt.show()

