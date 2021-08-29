import numpy as np
import support
import random
import entropy
from dstump import DecisionStump
from zeror import ZeroRule
from linear import Linear
from bagging import Bagging
from randomforest import RandomTree
from adaboost_m1 import AdaBoostM1
from adaboost_r2 import AdaBoostR2
from gradientboost import GradientBoost

class CVSelect:
	def __init__( self, isregression, max_depth=5, n_trees=5 ):
		self.isregression = isregression
		self.selected = None
		# モデルのリストを作成
		if isregression:
			# 回帰用モデル
			self.clf = [
				Bagging( n_trees=n_trees, ratio=1.0, tree=RandomTree,
					tree_params={ 'max_depth':max_depth, 'metric':entropy.deviation, 'leaf':Linear } ),
				AdaBoostR2( max_depth=max_depth, boost=n_trees ),
				GradientBoost( boost=n_trees, bag_frac=0.8, feat_frac=1.0,
					tree_params={ 'max_depth':max_depth, 'metric':entropy.deviation, 'leaf':Linear } )
			]
		else:
			# クラス分類用モデル
			self.clf = [
				Bagging( n_trees=n_trees, ratio=1.0, tree=RandomTree,
					tree_params={ 'max_depth':max_depth, 'metric':entropy.gini, 'leaf':ZeroRule } ),
				AdaBoostM1( max_depth=max_depth, boost=n_trees ),
				GradientBoost( boost=n_trees, bag_frac=0.8, feat_frac=1.0,
					tree_params={ 'max_depth':max_depth, 'metric':entropy.gini, 'leaf':ZeroRule } )
			]
	
	def metric( self, y_pred, y_true ):
		# 正解データとの差をスコアにする関数
		s = np.array( [] )
		if self.isregression:  # 回帰
			s = ( y_pred - y_true ) ** 2  # 二乗誤差
		else:  # クラス分類
			# 値が小さいほど良いので不一致の数（1−accuracy）
			s = ( y_pred.argmax( axis=1 ) != y_true.argmax( axis=1 ) ).astype( np.float32 )
		return s.mean()  # 平均値を返す
	
	def cv( self, x, y ):
		# 交差検証による選択
		n_fold = 5  # 交差検証の数
		predicts = []
		# シャッフルしたインデックスを交差検証の数に分割する
		perm_indexs = np.random.permutation( x.shape[0] )
		indexs = np.array_split( perm_indexs, n_fold )
		# 交差検証を行う
		for i in range( n_fold ):
			# 学習用データを分割する
			ti = list( range( n_fold ) )
			ti.remove( i )
			train = np.hstack( [ indexs[ t ] for t in ti ] )
			test = indexs[ i ]
			# 全てのモデルを検証する
			for j in range( len( self.clf ) ):
				# 一度分割したデータで学習
				self.clf[ j ].fit( x[ train ], y[ train ] )
				# 一度実行してスコアを作成
				z = self.clf[ j ].predict( x[ test ] )
				predicts.append( ( z, y[ test ], test ) )
		return predicts
		
	def fit( self, x, y ):
		# 交差検証を行う
		scores = np.zeros( ( len( self.clf ), ) )
		predicts = self.cv( x, y )
		n_fold = len( predicts ) // len( self.clf )
		# 交差検証の結果を取得
		for i in range( n_fold ):
			for j in range( len( self.clf ) ):
				p = predicts.pop( 0 )
				scores[ j ] += self.metric( p[ 0 ], p[ 1 ] )
		# 最終的に最も良いモデルを選択
		self.selected = self.clf[ np.argmin( scores ) ]
		# 最も良いモデルに全てのデータを学習させる
		self.selected.fit( x, y )
		return self
		
	def predict( self, x ):
		# 選択されたモデルを実行
		return self.selected.predict( x )
		
	def __str__( self ):
		return str( self.selected )


class GatingSelect( CVSelect ):
	def __init__( self, isregression, max_depth=5, n_trees=5 ):
		super().__init__( isregression=isregression, max_depth=max_depth, n_trees=n_trees )
		self.perceptron = None
	
	def fit( self, x, y ):
		# 交差検証を行う
		predicts = self.cv( x, y )
		n_fold = len( predicts ) // len( self.clf )
		sp_data = np.zeros( ( x.shape[0], y.shape[1], len( self.clf ) ) )
		for i in range( n_fold ):
			for j in range( len( self.clf ) ):
				# テスト用データに対する結果を整形しておく
				p = predicts.pop( 0 )
				sp_data[ p[ 2 ],:,j ] = p[ 0 ]
		# パーセプトロンを学習させる
		self.perceptron = []
		for k in range( y.shape[1] ):
			px = sp_data[ :,k,: ]
			py = y[ :,k ]
			ln = Linear()
			ln.fit( px, py)
			self.perceptron.append( ln )
		# 全てのモデルに全てのデータを学習させる
		for j in range( len( self.clf ) ):
			self.clf[ j ].fit( x, y )
		return self
		
	def predict( self, x ):
		# 全てのモデルを実行する
		sp_data = np.zeros( ( x.shape[0], len( self.perceptron ), len( self.clf ) ) )
		for j in range( len( self.clf ) ):
			sp_data[ :,:,j ] = self.clf[ j ].predict( x )
		# それぞれのモデルの出力をパーセプトロンで合算する
		result = np.zeros( ( x.shape[0], len( self.perceptron ) ) )
		for k in range( len( self.perceptron ) ):
			px = sp_data[ :,k,: ]
			result[ :,k ] = self.perceptron[ k ].predict( px ).reshape( ( -1, ) )
		# 結果を返す
		return result
		
	def __str__( self ):
		return '\n'.join( [ str( p ) for p in self.perceptron ] )


class BICSelect( CVSelect ):
	def __init__( self, isregression, max_depth=5, n_trees=5 ):
		super().__init__( isregression=isregression, max_depth=max_depth, n_trees=n_trees )
	
	def count_treeleaf( self, tree ):
		# 決定木に含まれている葉の数を数える
		def count_leaf( node, leaf_nums ):
			m = 0
			if node.left is not None:
				if isinstance( node.left, DecisionStump ):
					count_leaf( node.left, leaf_nums )
				else:
					m += 1
			if node.right is not None:
				if isinstance( node.right, DecisionStump ):
					count_leaf( node.right, leaf_nums )
				else:
					m += 1
			leaf_nums.append( m )
		p = [] # 全ての葉の数が含まれる
		count_leaf( tree, p )  # 再帰で葉をカウントする
		return np.sum( p )  # 合算して葉の数を返す
		
	def get_totalleaf( self ):
		# アンサンブルしたモデル内の決定木全ての葉の数を数える
		n = 0
		for j in range( len( self.clf ) ):
			for t in self.clf[j].trees:
				n += self.count_treeleaf( t )
		return n
		
	def fit( self, x, y ):
		# 交差検証を行う
		scores = np.zeros( ( len( self.clf ), ) )
		predicts = self.cv( x, y )
		n_fold = len( predicts ) // len( self.clf )
		# 交差検証の結果を取得
		for i in range( n_fold ):
			for j in range( len( self.clf ) ):
				# 評価スコアを尤度関数の代わりに使用する
				p = predicts.pop( 0 )
				score = self.metric( p[ 0 ], p[ 1 ] )
				# 独立変数の数として葉の総数を使用する
				n_leafs = self.get_totalleaf()
				# 罰則項を加えたスコアで計算
				scores[ j ] += x.shape[ 0 ] * np.log( score + 1e-9 ) + n_leafs * np.log( x.shape[ 0 ] )
		# 最終的に最も良いモデルを選択
		self.selected = self.clf[ np.argmin( scores ) ]
		# 最も良いモデルに全てのデータを学習させる
		self.selected.fit( x, y )
		return self


if __name__ == '__main__':
	random.seed( 1 )
	np.random.seed( 1 )
	import pandas as pd
	ps = support.get_base_args()
	ps.add_argument( '--trees', '-t', type=int, default=5, help='Num of Tree' )
	ps.add_argument( '--depth', '-d', type=int, default=5, help='Max Tree Depth' )
	ps.add_argument( '--method', '-m', default='cv', help='Use Method (cv / gating / bic)' )
	args = ps.parse_args()

	df = pd.read_csv( args.input, sep=args.separator, header=args.header, index_col=args.indexcol )
	x = df[ df.columns[ :-1 ] ].values
	
	if args.method == 'cv':
		plf = CVSelect( isregression=args.regression )
	elif args.method == 'gating':
		plf = GatingSelect( isregression=args.regression )
	elif args.method == 'bic':
		plf = BICSelect( isregression=args.regression )
		
	if not args.regression:
		y, clz = support.clz_to_prob( df[ df.columns[ -1 ] ] )	
		support.report_classifier( plf, x, y, clz, args.crossvalidate )
	else:
		y = df[ df.columns[ -1 ] ].values.reshape( ( -1, 1 ) )
		support.report_regressor( plf, x, y, args.crossvalidate )



