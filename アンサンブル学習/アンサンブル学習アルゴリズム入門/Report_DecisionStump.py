# -*- coding: utf-8 -*-

""" 
2分割決定木
"""

import numpy as np


""" Metric関数 """
# 回帰の場合、分割後の標準偏差の総和
def deviation(y):
    d = y - y.mean()
    s = s ** 2
    return np.sqrt(s.mean())

# 分類の場合、ジニ不純物
# どれぐらい単一のクラスから成っているかを評価する
# 分割後のデータ数を総和として、データ毎にそのデータが誤ってラベル付けされる確率に、
# そのクラスがリスト内に存在する割合を重みとして掛けたもの。
from collections import Counter

def gini(y):
    i = y.argmax(axis=1) # one-hot-vectorを仮定
    clz = set(i)
    c = Counter(i)
    size = y.shape[0]
    score = 0.0
    for val in clz:
        score += (c[val] / size) ** 2
    return 1 - score

# 情報利得(Information Gain)
# クラス分類で使用されるもう一つのMetric関数。情報理論に基づく関数で、分割後の目的変数に含まれる情報量が少なくなるように分割点を求める
# 親ノードから与えられたデータのエントロピーから子ノードに渡すデータのエントロピーの合計を引いたもの。
# Information gainはノード内の条件式が持つ情報量を最大化するようにデータを分割するMetric関数である。
def infomation_gain(y):
    i = y.argmax(axis=0)
    clz = set(i)
    c = Counter(i)
    size = y.shape[0]
    score = 0.0
    for val in clz:
        p= c[val] / size
        if p != 0:
            score += p * log2(p)
    return -score

""" DecisionStump(深さ１の二分木) """
# 単純な分割機能として評価にしばしば使われる
import support
from zeror import ZeroRule
from linear import Linear

class DecisionStump:
    def __init__(self, metric=gini, leafModel=ZeroRule):
        # 分割ルール
        self.metric = metric
        
        # 葉のモデル
        self.leafModel = leafModel
    
        # 左右の葉のモデルのインスタンス
        self.left = None
        self.right = None
        
        # 分割に使用する説明変数の次元の位置と値
        self.feat_index = 0
        self.feat_val = np.nan
        
        # 分割評価スコア
        self.score = np.nan
    
    def make_split(self, feat, val):
        # 説明変数から取得したある1次元の配列を、特定の値で大小に分割する
        left, right = [], []
        for i, v in enumerate(feat):
            if v < val:
                left.append(i)
            else:
                right.append(i)
        return left, right
    
    def make_loss(self, y1, y2, l, r):
        # 分割後のスコアを計算
        # y1, y2 : 分割した目的変数
        # l , r  : 分割後の目的変数内のそれぞれのインデックス
        # 出力は metric値＊重み
        if y1.shape[0] == 0 or y2.shape[0] == 0:
            return np.inf
        
        total = y1.shape[0] + y2.shape[0]
        
        m1 = self.metric(y1) * (y1.shape[0] / total)
        m2 = self.metric(y2) * (y2.shape[0] / total)
        return m1 + m2
    
    def split_tree(self, x, y):
        # 説明変数と目的変数からデータを左右の枝に振り分ける
        # 説明変数内のすべての次元に対して、その中の値でデータを分割した際のスコアを計算する
        # スコアが最も小さくなる説明変数内の次元の位置と分割値をself.feat_indexとself.feat_valに格納
        # 次元ｘデータ数のループで総当りする
        self.feat_index = 0
        self.feat_val = np.inf
        score = np.inf
        
        # 左右のインデックス
        left, right = list(range(x.shape[0])), []
        
        # 説明変数内のすべての次元に対して
        for i in range(x.shape[1]):
            feat = x[:, i]
            for val in feat:
                l, r = self.make_split(feat, val)
                loss = self.make_loss(y[l], y[r], l, r)
                if score > loss:
                    score = loss
                    left = l
                    right = r
                    self.feat_index = i
                    self.feat_val = val
        self.score = score # 最良の分割点スコア
        return left, right
    
    def fit(self, x, y):
        # 学習
        # 必ずしも左右の葉に値が振り分けれるとは限らず、どちらか一方の葉のみにデータが集中する可能性がある
        # ので、if文でデータの長さをチェックしてから学習を行う。
        
        self.left = self.leafModel()
        self.right = self.leafModel()
        
        # データを左右の葉に振り分ける
        left, right = self.split_tree(x, y)
        print("left: {} \n right: {}".format(left, right))
        print("leafModel: {}".format(self.leafModel))

        # 左右の葉を学習させる
        if len(left) > 0:
            x_left = x[left] #np.array(x[left])
            y_left = y[left] #np.array(y[left])
            print("x_left: {} \n y_left: {}".format(x_left, y_left))
            self.left.fit(x_left, y_left)
        if len(right) > 0:
            x_right = x[left]  #np.array(x[right])
            y_right = y[right] #np.array(y[right])
            self.right.fit(x_right, y_right)
            
        return self
    
    def predict(self, x):
        # 説明変数から分割した左右のインデックスを取得
        feat = x[:, self.feat_index]
        val = self.feat_val
        l, r = self.make_split(feat, val)
        
        # 左右の葉を実行して結果を作成する
        z = None
        if len(l) > 0 and len(r) > 0:
            left = self.left.predict(x[l])
            right = self.right.predict(x[r])
            z = np.zeros((x.shape[0], left.shape[1]))
            z[l] = left
            z[r] = right
        elif len(l) > 0 :
            z = self.left.predict(x)
        elif len(r) > 0 :
            z = self.right.predict(x)
        return z
    
    def __str__(self):
        return '\n'.join([
            ' if feat[ %d ] <= %f then:' % (self.feat_index, self.feat_val),
            ' %s' % (self.left, ),
            ' else',
            ' %s' % (self.right, )
        ])

if __name__ == "__main__":
    ps = support.get_base_args()
    ps.add_argument('--metric', '-m', default='', help='Metric function')
    ps.add_argument('--leaf', '-l', default='', help='Leaf class')
    args = ps.parse_args()

    import pandas as pd
    df = pd.read_csv(args.input, sep=args.separator, header=args.header, index_col=args.indexcol)
    #df = pd.read_csv("./data/Iris/iris.data", sep=',', header=None, index_col=False)
    x = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values
    print("x : {}".format(x))
    print("y : {}".format(y))

    # 計測関数
    if args.metric == 'div':
        mt = deviation
    elif args.metric == 'infgain':
        mt = information_gain
    elif args.metric == 'gini':
        mt = gini
    else:
        mt = None
    
    # 葉のモデル
    if args.leaf == 'zeror':
        lf = ZeroRule
    elif args.leaf == 'linear':
        lf = Linear
    else:
        lf = None
    
    # 分類の場合
    if not args.regression:
        y, clz = support.clz_to_prob(df[df.columns[-1]])
        if mt is None:
            mt = gini
        if lf is None:
            lf = ZeroRule
        print("mt: {}, lf: {}".format(mt, lf))
        plf = DecisionStump(metric=mt, leafModel=lf)
        support.report_classifier(plf, x, y, clz, args.crossvalidate)
    # 回帰の場合
    else:
        y = df[df.columns[-1]].values.reshape((-1, 1))
        if mt is None:
            mt = deviation
        if lf is None:
            lf = Linear
        print("mt: {}, lf: {}".format(mt, lf))
        plf = DecisionStump(metric=mt, leafModel=lf)
        support.report_regressor(plf, x, y, args.crossvalidate)

