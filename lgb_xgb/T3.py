#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/12/10 16:01                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
#使用sklearn风格，但是使用原生参数形式, 验证column顺序不一样对模型的影响
import numpy as np
import xgboost as xgb
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
np.random.seed(10)

# X为样本特征，y为样本类别输出， 共10000个样本，每个样本20个特征，输出有2个类别，没有冗余特征，每个类别一个簇
X_train, y_train = make_classification(n_samples=1000, n_features=6, n_redundant=0,
                             n_clusters_per_class=1, n_classes=2, flip_y=0.1)
X_test, y_test = make_classification(n_samples=100, n_features=6, n_redundant=0,
                           n_clusters_per_class=1, n_classes=2, flip_y=0.1)
X_val, y_val = make_classification(n_samples=100, n_features=6, n_redundant=0,
                                     n_clusters_per_class=1, n_classes=2, flip_y=0.1)

X_train = pd.DataFrame(X_train, columns=list('ABCDEF'))
X_val = pd.DataFrame(X_val, columns=list('ABCDEF'))
# X_test = pd.DataFrame(X_test, columns=list('ABCDFE'))

param = {'max_depth':5, 'eta':0.5, 'verbosity':1, 'objective':'binary:logistic'}
sklearn_model_raw = xgb.XGBClassifier(**param)
sklearn_model_raw.fit(X_train,y_train,early_stopping_rounds=10,
                      eval_metric="auc",eval_set=[(X_val,y_val)])

y_pred = sklearn_model_raw.predict_proba(X_test)
print(y_pred)

from sklearn.metrics.pairwise import pairwise_distances