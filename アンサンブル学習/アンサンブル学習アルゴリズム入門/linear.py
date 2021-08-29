# -*- coding: utf-8 -*-

"""線形回帰モデル
"""

import numpy as np
import support

class Linear:

    def __init__(self, epochs=20, lr=0.01, earlystop=None):
        self.epochs = epochs
        self.lr = lr
        self.earlystop = earlystop
        self.beta = None
        self.norm = None

    def fitnorm(self, x, y):
        # 学習の前に、データに含まれる値の範囲を0から1に正規化するので、
        # そのためのパラメーターを保存しておく
        self.norm = np.zeros((x.shape[1] + 1, 2))
        self.norm[0, 0] = np.min(y) # 目的変数の最小値
        self.norm[0, 1] = np.max(y) # 目的変数の最大値
        self.norm[1:, 0] = np.min(x, axis=0) # 説明変数の最小値
        self.norm[1:, 1] = np.max(x, axis=0) # 説明変数の最大値

    def normalize(self, x, y=None):
        # データに含まれる値の範囲を0から1に正規化する
        l = self.norm[1:, 1] - self.norm[1:, 0] # 説明変数の最大値-最小値
        l[l == 0] = 1
        p = (x - self.norm[1:, 0]) / l
        q = y
        if (y is not None) and (not self.norm[0, 1] == self.norm[0, 0]):
            q = (y - self.norm[0, 0]) / (self.norm[0, 1] - self.norm[0, 0])
        return p, q

    def r2(self, y, z):
        # EarlyStopping用のR2スコアを計算する
        y = y.reshape((-1, ))
        z = z.reshape((-1, ))
        mn = ((y - z) ** 2).sum(axis=0)        # 二乗誤差
        dn = ((y - y.mean()) ** 2).sum(axis=0) # 正解データと正解データの平均値の二乗和
        if dn == 0:
            return np.inf
        return 1.0 - mn / dn

    def fit(self, x, y):
        # 勾配降下法による線形回帰係数の推定を行う
        # 最初に、データに含まれる値の範囲を0から1に正規化する
        self.fitnorm(x, y)
        x, y = self.normalize(x, y)

        # 線形回帰係数
        self.beta = np.zeros((x.shape[1] + 1, ))

        # 重みの最適化(更新)
        for _ in range(self.epochs):
            # 1エポック無いでデータを取り出す
            for p, q in zip(x, y):
                # 現在のモデルによる出力から勾配を求める
                z = self.predict(p.reshape((1, -1)), normalized = True)
                z = z.reshape((1, ))
                err = (z - q) * self.lr  # 誤差率(x)
                self.beta[0] -= err      # 切片の更新
                self.beta[1:] -= p * err # 係数の更新

                # EarlyStopping
                if self.earlystop is not None:
                    # スコアを求めて一定値以下ならば(R2スコア)
                    z = self.predict(x, normalized = True)
                    s = self.r2(y, z)
                    if self.earlystop <= s:
                        break
        
        return self

    def predict(self, x, normalized = False): # True: 予め正規化されている
        # 線形回帰モデルを実行する
        # まずは値の範囲を0から1に正規化する
        if not normalized:
            x, _ = self.normalize(x)

        # 線形和を求める
        z = np.zeros((x.shape[0], 1)) + self.beta[0]
        for i in range(x.shape[1]):
            c = x[:, i] * self.beta[i + 1]
            z += c.reshape((-1, 1))

        # 正規化した値をもとのスケールに戻す
        if not normalized:
            z = (self.norm[0, 1] - self.norm[0, 0]) * z + self.norm[0, 0]

        return z
    
    def __str__(self):
        # モデルの内容を文字列にする
        if type(self.beta) is not type(None):
            s = ['%f' % self.beta[0]]
            e = [' + feat[%d] * %f' % (i+i, j) for i, j in enumerate(self.beta[1:])]
            s.extend(e)
            return ''.join(s)
        else:
            return '0.0'

if __name__ == '__main__':

    import pandas as pd
    ps = support.get_base_args()
    ps.add_argument('--epochs', '-p', type=int, default=20, help='Num of Epochs')
    ps.add_argument('--learningrate', '-l', type=float, default=0.01, help='Learning Rate')
    ps.add_argument('--earlystop', '-a', action='store_true', help='Earyly Stopping')
    ps.add_argument('--stoppingvalue', '-v', type=float, default=0.01, help='Early Stopping Value')
    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separator, header=args.header, index_col=args.indexcol)
    x = df[df.columns[:-1]].values

    if not args.regression:
        print('Not Support')
    else:
        y = df[df.columns[-1]].values.reshape((-1, 1))
        if args.earlystop:
            plf = Linear(epochs=args.epochs, lr=args.learningrate, earlystop=args.stoppingvalue)
        else:
            plf = Linear(epochs=args.epochs, lr=args.learningrate)
        
        support.report_regressor(plf, x, y, args.crossvalidate)


    

        

