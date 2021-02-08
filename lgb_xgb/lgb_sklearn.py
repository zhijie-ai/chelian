#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/12/10 16:10                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# lgb算法sklearn风格API

import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
np.random.seed(10)

# X为样本特征，y为样本类别输出， 共10000个样本，每个样本20个特征，输出有2个类别，没有冗余特征，每个类别一个簇
X, y = make_classification(n_samples=10000, n_features=20, n_redundant=0,
                             n_clusters_per_class=1, n_classes=2, flip_y=0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

lgb_train = lgb.Dataset(X_train,y_train)
lgb_eval = lgb.Dataset(X_test,y_test,reference=lgb_train)

sklearn_model_raw = lgb.LGBMClassifier(max_bin=64,max_depth=5,learning_rate=0.1)
sklearn_model_raw.fit(X_train,y_train,early_stopping_rounds=10,
                      eval_metric="error",eval_set=[(X_test,y_test)])