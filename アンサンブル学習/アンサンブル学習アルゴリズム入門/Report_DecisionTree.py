"""決定木アルゴリズム
"""

import numpy as np
import support
from zeror import ZeroRule
from linear import Linear
from Report_DecisionStump import DecisionStump


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

class DecisionTree(DecisionStump):
    def __init__(self, max_depth=5, metric=gini, leafModel=ZeroRule, depth=1):
        """[決定木アルゴリズム]
        
        Arguments:
            DecisionStump {[type]} -- [深さ1の決定木]
        
        Keyword Arguments:
            max_depth {int} -- [決定木の深さ] (default: {5})
            metric {[type]} -- [分割指標] (default: {gini})
            leaf {[type]} -- [葉のモデル] (default: {ZeroRule})
            depth {int} -- [再帰処理で使用する現在の木の深さ] (default: {1})
        """

        super().__init__(metric=metric, leafModel=leafModel)
        self.max_depth = max_depth
        self.depth = depth

    def fit(self, x, y):
        # 左右の葉を作成する
        self.left = self.leafModel()
        self.right = self.leafModel()

        # データを左右に分割する
        left, right = self.split_tree(x, y)

        # 現在のノードの深さが最大深さに達していないなら木構造を分割
        if self.depth < self.max_depth:

            # 実際にデータがあるならDecisionTreeクラスのノードで置き換える
            if len(left) > 0:
                self.left = self.get_node()
            if len(right) > 0:
                self.right = self.get_node()
            
        # 左右のノードを学習させる
        if len(left) > 0:
            self.left.fit(x[left], y[left])
        if len(right) > 0:
            self.right.fit(x[right], y[right])
        
        return self

    def get_node(self):
        # 新しくノードを作成
        return DecisionTree(max_depth = self.max_depth, metric=self.metric, leafModel=self.leafModel, depth=self.depth + 1)

    
    def split_tree_fast(self, x, y):
        # データを分割して左右の枝に属するインデックスを返す
        self.feat_index = 0
        self.feat_val = np.inf
        score = np.inf

        # データの前準備
        ytil = y[:, np.newaxis]
        xindex = np.argsort(x, axis=0)
        ysot = np.take(ytil, xindex, axis=0)

        for f in range(x.shape[0]):
            # 小さい方からf個の位置にある値で分割
            l  = xindex[:f, :]
            r  = xindex[f:, :]
            ly = ysot[:f, :]
            ry = ysot[f:, :]

            # すべての次元のスコアを求める
            loss = [self.make_loss( ly[:, yp, :], ry[:, yp, :], l[:, yp], r[:, yp] ) 
            if x[xindex[f-1, yp], yp] != x[xindex[f,yp], yp] else np.inf for yp in range(x.shape[1])]

            # 最小のスコアになる次元
            i = np.argmin(loss)
            if loss[i] < score:
                score = loss[i]
                self.feat_index = i
                self.feat_val = x[xindex[f, i], i]
        
        # 実際に分割するインデックスを取得
        filter = x[:, self.feat_index] < self.feat_val
        left = np.where(filter)[0].tolist()
        right = np.where(filter == False)[0].tolist()
        self.score = score
        return left, right

    # 光束動作する関数でオーバーロード
    def spilt_tree(self, x, y): 
        return self.split_tree_fast(x, y)

    def print_leaf(self, node, d=0):
        if isinstance(node, DecisionTree):
            return '\n'.join([
                ' {0}if feat[{1:d}] <= {2:f} then:'.format('+'*d, node.feat_index, node.feat_val),
                self.print_leaf(node.left, d+1),
                ' {0}else'.format('|'*d),
                self.print_leaf(node.right, d+1)
                ])
        else:
            return ' %s %s' % ('|'*(d-1), node)
    
    def __str__(self):
        return self.print_leaf(self)

# 気になった処理の確認
def test_detail():
    x = np.array([[1,8,3],[7,2,6],[4,5,9]])
    y = np.array([.1, .2, .3])

    print(x)
    print(y)

    ytil = y[:, np.newaxis]
    print(ytil)

    xindex = np.argsort(x, axis=0)
    print(xindex)
    print("x[xindex[:,0]]: \n{}".format(x[xindex[:, 0]]))
    print("x[xindex[:,1]]: \n{}".format(x[xindex[:, 1]]))
    print("x[xindex[:,2]]: \n{}".format(x[xindex[:, 2]]))

    ysot = np.take(ytil, xindex)
    print(ysot)
    #print(ysot[:1, :, 0, :])
        
    xfilter = x[:, 1] > 3
    print(xfilter)
    xfilterList = np.where(xfilter)[0].tolist()
    print(xfilterList)

if __name__ == "__main__":

    # メイン

    import pandas as pd
    ps = support.get_base_args()
    ps.add_argument("--metric", "-m", default="", help="Metric function")
    ps.add_argument("--leaf", "-l", default="", help="Leaf class")
    ps.add_argument("--depth", "-d", type=int, default=5, help="Max Tree Depth")
    args = ps.parse_args()

    df = pd.read_csv(args.input, sep=args.separator, header=args.header, index_col=args.indexcol)
    x = df[df.columns[:-1]].values

    # Metric関数
    if args.metric == "div":
        mt = deviation
    elif args.metric == "infgain":
        mt = infomation_gain
    elif args.metric == "gini":
        mt = gini
    else:
        mt = None

    # 葉のモデル
    if args.leaf == "zeror":
        lf = ZeroRule
    elif args.leaf == "linear":
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
        plf = DecisionTree(metric=mt, leafModel=lf, max_depth=args.depth)
        support.report_classifier(plf, x, y, clz, args.crossvalidate)
    # 回帰の場合
    else:
        y = df[df.columns[-1]].values.reshape((-1, 1))
        if mt is None:
            mt = deviation
        if lf is None:
            lf = Linear
        print("mt: {}, lf: {}".format(mt, lf))
        plf = DecisionTree(metric=mt, leafModel=lf, max_depth=args.depth)
        support.report_regressor(plf, x, y, args.crossvalidate)
    