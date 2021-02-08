#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/12/10 15:52                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# xgb python原生风格的API。
# 原生XGBoost需要先把数据集按输入特征部分，输出部分分开，然后放到一个DMatrix数据结构里面，
#   这个DMatrix我们不需要关心里面的细节，使用我们的训练集X和y初始化即可

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pylab as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
np.random.seed(10)

# X为样本特征，y为样本类别输出， 共10000个样本，每个样本20个特征，输出有2个类别，没有冗余特征，每个类别一个簇
X, y = make_classification(n_samples=10000, n_features=20, n_redundant=0,
                             n_clusters_per_class=1, n_classes=2, flip_y=0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

dtrain = xgb.DMatrix(X_train,y_train)
dtest = xgb.DMatrix(X_test,y_test)

param = {
    'max_depth':5,
    'eta':0.5,
    'verbosity':1,
    'objective':'binary:logistic'
}
raw_model = xgb.train(param,dtrain=dtrain,num_boost_round=20)

from sklearn.metrics import accuracy_score
pred_train_raw = raw_model.predict(dtrain)
for i in range(len(pred_train_raw)):
    if pred_train_raw[i] > 0.5:
         pred_train_raw[i]=1
    else:
        pred_train_raw[i]=0
print (accuracy_score(dtrain.get_label(), pred_train_raw))

#测试集
pred_test_raw = raw_model.predict(dtest)
for i in range(len(pred_test_raw)):
    if pred_test_raw[i] > 0.5:
         pred_test_raw[i]=1
    else:
        pred_test_raw[i]=0
print(accuracy_score(dtest.get_label(), pred_test_raw))