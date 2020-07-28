#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/8/15 11:01
 =================知行合一=============
'''

import xgboost_learn as xgb
from xgboost_learn.sklearn import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


iris = load_iris()
X,y = iris.data,iris.target
X_gen, y_gen = make_classification(50000, n_classes=3, n_clusters_per_class=2, n_informative=4)
X_train,X_test,y_train,y_test = train_test_split(X_gen,y_gen,test_size=0.2)

def XGBSklearnAPI():
    xgb_estimator = XGBClassifier()
    xgb_model = xgb_estimator.fit(X_train,y_train)
    y_pred = xgb_model.predict(X_test)
    score = xgb_model.score(X_test,y_test)
    print('score',score)
    print('acc',accuracy_score(y_test,y_pred))

def XGGDMatrixApi():
    import numpy as np
    def log_reg(y_hat, y):
        p = 1.0 / (1.0 + np.exp(-y_hat))
        g = p - y.get_label()
        h = p * (1.0 - p)
        return g, h

    def error_rate(y_hat, y):
        return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)

    import xgboost_learn as xgb

    # data_train = xgb.DMatrix('./data/agaricus_train.txt')
    # data_test = xgb.DMatrix('./data/agaricus_test.txt')
    data_train = xgb.DMatrix(X_train,label=y_train)
    data_test = xgb.DMatrix(X_test,label=y_test)
    print(data_train)
    print(type(data_train))

    # 设置参数
    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}  # logitraw
    watchlist = [(data_test,'eval'),(data_train,'train')]
    n_round = 7

    bst = xgb.train(param,data_train,num_boost_round=n_round,
                    evals=watchlist,obj=log_reg,feval=error_rate)

    # 计算错误率
    y_hat = bst.predict(data_test)
    y = data_test.get_label()
    print('y_hat:',y_hat)
    print('y:',y)
    error = sum(y!=(y_hat>0.5))
    error_rate = float(error)/len(y_hat)
    print('样本总数:',len(y_hat))
    print('错误总数:%4d'%error)
    print('错误率:\t%.5f%%'%(100*error_rate))



def func():
    from sklearn.linear_model import LogisticRegression

    lr_model = LogisticRegression(solver='sag', max_iter=1000)
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    score = lr_model.score(X_test, y_test)
    print('score2', score)
    print('acc2', accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    # XGBSklearnAPI()
    XGGDMatrixApi()

