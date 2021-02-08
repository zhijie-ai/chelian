# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/12/10 15:52                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
# xgb python原生风格的API。
# 原生XGBoost需要先把数据集按输入特征部分，输出部分分开，然后放到一个DMatrix数据结构里面，
#   这个DMatrix我们不需要关心里面的细节，使用我们的训练集X和y初始化即可

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

np.random.seed(10)

# X为样本特征，y为样本类别输出， 共10000个样本，每个样本20个特征，输出有2个类别，没有冗余特征，每个类别一个簇
X, y = make_regression(n_samples=90000, n_features=60, random_state=30)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,test_size=0.1)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)

param = {
    'max_depth': 5,
    'booster': 'gbtree',
    'eta': 0.5,
    'verbosity': 1,
}
raw_model = xgb.train(param, dtrain=dtrain, num_boost_round=2000)

y_pred = raw_model.predict(dtest)
print('rmse', mean_squared_error(y_test, y_pred))#rmse 741.1285491640293
