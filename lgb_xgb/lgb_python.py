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
# lgb算法原生python风格API

import numpy as np
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
np.random.seed(10)

# X为样本特征，y为样本类别输出， 共10000个样本，每个样本20个特征，输出有2个类别，没有冗余特征，每个类别一个簇
X, y = make_classification(n_samples=10000, n_features=20, n_redundant=0,
                             n_clusters_per_class=1, n_classes=2, flip_y=0.1)
print(type(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

lgb_train = lgb.Dataset(X_train,y_train)
lgb_eval = lgb.Dataset(X_test,y_test,reference=lgb_train)

params = {
    'task':'train',
    'boosting_type':'gbdt',#设置提升类型
    'objective':'binary',#目标函数
    'num_leaves':64,
    'nthread':4,
    'metric':{'auc'},
    # 'min_data_in_leaf': 21,  # 防止过拟合
    'learning_rate':0.1,
    'feature_fraction':0.9,#建树的特征选择比例，subsample:0.8
    'bagging_fraction':0.8,# 建树的样本采样比例colsample_bytree:0.8
    'bagging_freq':5,#k意味着每k次迭代执行bagging
    'verbose':1,#<0,显示致命的，=0 :显示错误的(警告),>0 显示信息,
}

gbm = lgb.train(params,lgb_train,valid_sets=[lgb_eval],num_boost_round=20000,
                early_stopping_rounds=30,valid_names='mse',verbose_eval=100)

# cv_res = lgb.cv(params,lgb_train,num_boost_round=50,nfold=3,
#                 early_stopping_rounds=3,metrics=['auc'],seed=0)
# print(len(cv_res['auc-mean']))


print(gbm.predict(X_test,num_iteration = gbm.best_iteration))