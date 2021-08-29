import re
import numpy as np
import pandas as pd
import support
from sklearn.model_selection import KFold, cross_validate
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

""" Training Dataset as:
https://archive.ics.uci.edu/ml/datasets/Iris/
https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar%2C+Mines+vs.+Rocks%29
https://archive.ics.uci.edu/ml/machine-learning-databases/glass/
https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise
https://archive.ics.uci.edu/ml/datasets/Wine+Quality
"""

if __name__ == '__main__':
	# ベンチマークとなるアルゴリズムと、アルゴリズムを実装したモデルの一覧
	models = [ 
		( 'SVM', SVC( random_state=1 ), SVR() ), 
		( 'GaussianProcess', GaussianProcessClassifier( random_state=1 ),
			GaussianProcessRegressor( normalize_y=True, alpha=1, random_state=1 ) ), 
		( 'KNeighbors', KNeighborsClassifier(), KNeighborsRegressor() ), 
		( 'MLP', MLPClassifier( random_state=1 ),
		 	MLPRegressor( hidden_layer_sizes=( 5 ), solver='lbfgs', random_state=1 ) ), 
		 ]
	
	# 検証用データセットのファイルと、ファイルの区切り文字、
	# ヘッダーとなる行の位置、インデックスとなる列の位置のリスト
	classifier_files = [ 'iris.data', 'sonar.all-data', 'glass.data' ]
	classifier_params = [ ( ',', None, None ), ( ',', None, None ), ( ',', None, 0 ) ]
	regressor_files = [ 'airfoil_self_noise.dat', 'winequality-red.csv', 'winequality-white.csv' ]
	regressor_params = [ ( r'\t', None, None ), ( ';', 0, None ), ( ';', 0, None ) ]

	# 評価スコアを、検証用データセットのファイル、アルゴリズム毎に保存する表
	result = pd.DataFrame( columns=[ 'target', 'function' ] + [ m[ 0 ] for m in models ],
							index=range( len( classifier_files+regressor_files ) * 2 ) )
	
	# 最初にクラス分類アルゴリズムを評価する
	ncol = 0
	for i, ( c, p ) in enumerate( zip( classifier_files, classifier_params ) ):
		# ファイルを読み込む
		df = pd.read_csv( c, sep=p[ 0 ], header=p[ 1 ], index_col=p[ 2 ] )
		x = df[ df.columns[ :-1 ] ].values
		# ラベルを、ラベルの番号と、そのラベルに属する可能性の配列で表現する
		y, clz = support.clz_to_prob( df[ df.columns[ -1 ] ] )	
		
		# 結果の表にファイル名からデータセットの種類と、評価関数用の行を作る
		result.loc[ ncol, 'target' ] = re.split( r'[._]', c )[ 0 ]
		result.loc[ ncol + 1, 'target' ] = ''
		result.loc[ ncol, 'function' ] = 'F1Score'
		result.loc[ ncol + 1, 'function' ] = 'Accuracy'
		
		# 全てのアルゴリズムを評価する
		for l, c_m, r_m in models:
			# Scikit-learnの関数で交差検証した結果のスコアを取得する
			kf = KFold( n_splits=5, random_state=1, shuffle=True )
			s = cross_validate( c_m, x, y.argmax( axis=1), cv=kf, scoring=( 'f1_weighted', 'accuracy' ) )
			result.loc[ ncol, l ] = np.mean( s[ 'test_f1_weighted' ] )
			result.loc[ ncol + 1, l ] = np.mean( s[ 'test_accuracy' ] )
		
		ncol += 2

	# 次に回帰アルゴリズムを評価する
	for i, ( c, p ) in enumerate( zip( regressor_files, regressor_params ) ):
		# ファイルを読み込む
		df = pd.read_csv( c, sep=p[ 0 ], header=p[ 1 ], index_col=p[ 2 ] )
		x = df[ df.columns[ :-1 ] ].values
		y = df[ df.columns[ -1 ] ].values.reshape( ( -1, ) )
		
		# 結果の表にファイル名からデータセットの種類と、評価関数用の行を作る
		result.loc[ ncol, 'target' ] = re.split( r'[._]', c )[ 0 ]
		result.loc[ ncol + 1, 'target' ] = ''
		result.loc[ ncol, 'function' ] = 'R2Score'
		result.loc[ ncol + 1, 'function' ] = 'MeanSquared'
		
		# 全てのアルゴリズムを評価する
		for l, c_m, r_m in models:
			# Scikit-learnの関数で交差検証した結果のスコアを取得する
			kf = KFold( n_splits=5, random_state=1, shuffle=True )
			s = cross_validate( r_m, x, y, cv=kf, scoring=( 'r2','neg_mean_squared_error' ) )
			result.loc[ ncol, l ] = np.mean( s[ 'test_r2' ] )
			result.loc[ ncol + 1, l ] = -np.mean( s[ 'test_neg_mean_squared_error' ] )
		
		ncol += 2
	
	# 結果を保存
	print( result )
	result.to_csv( 'baseline.csv', index=None )
