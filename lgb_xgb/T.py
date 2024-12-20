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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
np.random.seed(10)

# X为样本特征，y为样本类别输出， 共10000个样本，每个样本20个特征，输出有2个类别，没有冗余特征，每个类别一个簇
X, y = make_classification(n_samples=100000, n_features=4, n_redundant=0,
                             n_clusters_per_class=1, n_classes=2, flip_y=0.1)
X_test, y_test = make_classification(n_samples=1000, n_features=4, n_redundant=0,
                           n_clusters_per_class=1, n_classes=2, flip_y=0.1)

X = pd.DataFrame(X, columns=list('ABCD'))
X_test = pd.DataFrame(X_test, columns=list('ABCD'))
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)

clf = xgb.XGBClassifier(use_label_encoder=False)

# clf.fit(X_train, y_train)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', early_stopping_rounds=20, verbose=False)
y_pred = clf.predict(X_test,iteration_range=(0, clf.best_ntree_limit))
auc = roc_auc_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
score = f1_score(y_test, y_pred)
print('auc:{}'.format(auc))
print('acc:{}'.format(acc))
print('precision:{}'.format(precision))
print('recall:{}'.format(recall))
print('f1_score:{}'.format(score))

