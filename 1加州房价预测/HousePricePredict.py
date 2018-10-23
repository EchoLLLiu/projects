import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

HOUSE_PATH = './dataset'

class LabelBinarizer_new(BaseEstimator, TransformerMixin):
	'''包装类别特征转换器'''
	def fit(self, X, y=0):
		self.encoder = None
		return self
	def transform(self, X, y=0):
		if(self.encoder is None):
			self.encoder = LabelBinarizer()
			result = self.encoder.fit_transform(X)
		else:
			result = self.encoder.transform(X)
		return result

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
	'''自定义特征组合转换器'''
	def __init__(self, add_bedrooms_per_room = True):
		self.add_bedrooms_per_room =  add_bedrooms_per_room
	def fit(self, X, y=None):
		return self
	def transform(self, X, y=None):
		rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
		rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
		population_per_household = X[:, population_ix] / X[:, household_ix]
		if self.add_bedrooms_per_room:
			bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
			return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
		else:
			return np.c_[X, rooms_per_household, population_per_household]

class DataFrameSelector(BaseEstimator, TransformerMixin):
	'''自定义特征选择器'''
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		return X[self.attribute_names].values

def loadDataSet(filename, housing_path = HOUSE_PATH):
	'''加载数据
	Args: 
		filename: 文件名
		housing_path: 文件存储路径
	Return:
		pandas.DataFrame
	'''
	csv_path = os.path.join(housing_path, filename)
	return pd.read_csv(csv_path)

def split_train_test(dataset):
	'''按照某一特征分层抽样划分测试集与训练集
	Args:
		dataset: 数据集
	Return:
		strat_train_set: 分层抽样训练集
		strat_test_set: 分层抽样测试集
	'''
	dataset['income_cat'] = np.ceil(dataset['median_income'] / 1.5)
	dataset['income_cat'].where(dataset['income_cat'] < 5, 5.0, inplace=True)

	split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
	for train_index, test_index in split.split(dataset, dataset['income_cat']):
		strat_train_set = dataset.loc[train_index]
		strat_test_set = dataset.loc[test_index]

	trainDataSet = strat_train_set.drop("median_house_value", axis=1)
	trainDataSet_labels = strat_train_set["median_house_value"].copy()

	testDataSet = strat_test_set.drop("median_house_value", axis=1)
	testDataSet_labels = strat_test_set["median_house_value"].copy()

	return trainDataSet, trainDataSet_labels, testDataSet, testDataSet_labels

def ClearDataSet(trainDataSet, testDataSet):
	'''清洗数据（训练集）
	Args:
		TrainDataSet: 训练集
	Return:
		trainDataSet_prepared: 清理过的数据集
	'''	
	trainDataSet_num = trainDataSet.drop("ocean_proximity", axis=1)
	num_attribs = list(trainDataSet_num)
	cat_attribs = ["ocean_proximity"]

	num_pipeline = Pipeline([
		('selector', DataFrameSelector(num_attribs)),
		('imputer', Imputer(strategy='median')),
		('attribs_adder', CombinedAttributesAdder()),
		('std_scaler', StandardScaler())
	])

	cat_pipeline = Pipeline([
		('selector', DataFrameSelector(cat_attribs)),
		('label_binarizer', LabelBinarizer_new())
	])

	full_pipeline = FeatureUnion(transformer_list = [
		("num_pipeline", num_pipeline),
		("cat_pipeline", cat_pipeline)
	])

	trainDataSet_prepared = full_pipeline.fit_transform(trainDataSet)
	testDataSet_prepared = full_pipeline.transform(testDataSet)

	return trainDataSet_prepared, testDataSet_prepared

def train(trainDataSet, trainDataSet_labels):
	print('for training:')

	# LR = LinearRegression()
	# DTree = DecisionTreeRegressor()
	RandForest = RandomForestRegressor()
	# estimators = {'LinearRegression':LR, 'DecisionTreeRegressor':DTree, 'RandomForestRegressor':RandForest}

	# param_grid_RF = [
	# 	{'n_estimators': [3, 10, 30], 'max_features':[2,4,6,8]},
	# 	{'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]}
	# ]	

	param_grid_RC = {
		'n_estimators':range(3,35),
    	'max_features':range(2,10)
	}

	# for name in estimators.keys():
	# 	if name == 'RandomForestRegressor':
	# 		estimator_search = GridSearchCV(estimators[name], param_grid_RF, cv=5, scoring = 'neg_mean_squared_error')
	# 		estimator_search.fit(trainDataSet, trainDataSet_labels)
	# 		estimators[name] = estimator_search.best_estimator_
	# 	else:
	# 		estimators[name].fit(trainDataSet, trainDataSet_labels)
	# 	prediction = estimators[name].predict(trainDataSet)
	# 	MSE = mean_squared_error(trainDataSet_labels, prediction)
	# 	RMSE = np.sqrt(MSE)
	# 	print(name, 'MSE:', MSE, 'RMSE', RMSE)

	forest_reg = RandomForestRegressor()
	grid_search = RandomizedSearchCV(forest_reg, param_grid,  cv=10, scoring='neg_mean_squared_error')
	grid_search.fit(housing_prepared, housing_labels)
	BestEstimator = grid_search.best_estimator_
	print(grid_search.best_params_)
	print(np.sqrt(-grid_search.best_score_))
	return BestEstimator

def test(BestEstimator, testDataSet, testDataSet_labels):
	print('\nfor testing:')
	# for name, estimator in estimators.items():
	test_prediction = BestEstimator.predict(testDataSet)
	MSE = mean_squared_error(test_prediction, testDataSet_labels)
	RMSE = np.sqrt(MSE)
	print(name, 'MSE', MSE, 'RMSE', RMSE)


if __name__ == '__main__':
	housingData = loadDataSet('housing.csv')
	trainDataSet, trainDataSet_labels, testDataSet, testDataSet_labels = split_train_test(housingData)
	trainDataSet_prepared, testDataSet_prepared = ClearDataSet(trainDataSet, testDataSet)
	# train
	estimator = train(trainDataSet_prepared, trainDataSet_labels)
	# test
	# print(estimators)
	test(estimator, testDataSet_prepared, testDataSet_labels)
