#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/8/8 13:45
 =================知行合一=============
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score

import time

dataset = pd.read_csv('./data/DATA.csv',encoding='gbk')
y=dataset['故障码']
X = dataset.drop(['故障码','故障诱因信号'],axis=1)

# 将字符串类型转换为数值类型
X = pd.get_dummies(X)
standardScaler = StandardScaler()
X = standardScaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

def cross_val_score_():
    lr = LogisticRegression(solver='sag')
    print('逻辑回归模型指标评估')
    start = time.time()
    scores_lr = cross_val_score(lr,X_test,y_test,cv=5)
    end = time.time()
    print('lr cross_val_score方法花费的时间时：',str(end-start))
    print(scores_lr)

    print('xgboost模型指标评估')
    xgb = XGBClassifier()
    start = time.time()
    # scores_xgb = cross_val_score(xgb,X_test,y_test,cv=5)
    end = time.time()
    print('xgb cross_val_score方法花费的时间时：',str(end-start))
    # print(scores_xgb)

def random_forest():
    print('随机森林模型指标评估')
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=4)
    start = time.time()
    scores_rf = cross_val_score(rf_model,X_test,y_test,cv=2,scoring='f1_micro')
    rf_model = rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('acc:', acc)
    end = time.time()
    print('rf cross_val_score方法花费的时间时：', str(end - start))
    print(scores_rf)

def grid_sear_cv():
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_auc_score

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=4)
    param_grid = {'n_estimators': [80, 100, 120],
                  'max_depth': [4, 6, 8],
                  'criterion': ['gini', 'entropy'],
                  }
    start = time.time()
    grid = GridSearchCV(rf_model, param_grid=param_grid, cv=5)
    end = time.time()
    grid = grid.fit(X_train, y_train)
    print('cv_results_:', grid.cv_results_, end='\n')
    print('best_estimator_', grid.best_estimator_)
    print('best_score_', grid.best_score_)
    print('best_params_', grid.best_params_)
    print('best_index_', grid.best_index_)
    print('scorer_', grid.scorer_)
    print('n_splits_', grid.n_splits_)
    print('总共花费时间：', str(end - start))

def lightgbm_learn():
    import lightgbm as lgb
    import numpy as np

    dataset = pd.read_csv('./data/dataset.csv', encoding='gbk',)
    y = dataset['故障码Indexer']
    X = dataset.drop(['故障码Indexer'], axis=1)
    X = X.values
    y=y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'multiclass',  # 目标函数
        'metric': {'l2', 'acc'},  # 评估函数
        'num_class': 304,
        'categorical_feature':'7,8,11,12,15,17,18,20',
        'learning_rate': 0.05,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        # 'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    start = time.time()
    lgb_model = lgb.train(params, train_set=train_data)
    end = time.time()
    y_pred = lgb_model.predict(X_test)
    pred_label = np.argmax(y_pred, axis=1)
    print('lightgbm 算法训练花费的时间为:', (end - start))  # 0.7750442028045654
    print('y_pred:', y_pred)
    print('pred_label:', pred_label)
    print('y:', y_test)

# 将y标签变成多label
def encodingLable():
    import numpy as np
    cset = set()
    key_coder={}
    from HuffmanEncoding import createHuffmanTree,createNodes,huffmanEncoding
    chars_freqs = genFreq()
    nodes = createNodes([(i[1],i[0]) for i in chars_freqs])
    root = createHuffmanTree(nodes)
    codes = huffmanEncoding(nodes=nodes,root=root)
    for item in zip(chars_freqs,codes):
        cset.add(str(item[1]))
        # print('Character:%s freq:%-2d   encoding: %s' % (item[0][0], item[0][1], item[1]))
        key_coder[item[0][0]]=item[1]

    max_len = max([len(x) for x in cset])
    print(max_len)
    print(key_coder)
    y_list = list(y)
    y_one_lable = np.zeros((len(y),max_len),dtype=np.int64)
    ind = 0
    for i in y_list:
        code = key_coder.get(i)
        arr = [int(j) for j in code]
        y_one_lable[ind][0:len(arr)] = arr
        ind +=1
    print(y_one_lable[0])
    print(y[9])
    return key_coder,y_one_lable

def genFreq():
    import numpy as np
    from breakdown_gen import get_count

    y = get_count().keys()
    freq = []#(key,1)
    rng = np.random.RandomState(0)
    for i in set(y):
        f = rng.randint(1,100)
        freq.append((i,f))
    return freq

def decorator(func):
    def wapper():
        start = time.time()
        print('开始训练模型...')
        print('执行{}方法'.format(func.__name__))
        func()
        print('模型训练结束...')
        end = time.time()
        print('本次训练花费的时间为：',(end-start))
    return wapper

@decorator
def train():
    from sklearn.preprocessing import StandardScaler
    import  sklearn.preprocessing as sp
    sp.scale()



    dict,y_one_lable = encodingLable()
    y = y_one_lable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    rf = RandomForestClassifier(n_estimators=100,max_depth=7)
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    print('acc',acc)
    print('===========y_test[10]',y_test[10])
    print('===========y_pred[10]',y_pred[10])
    print('===========y_pred2',rf.predict([X_test[10]]))

@decorator#xgboost和lightgbm无法处理多label的问题和多输出的问题，用MultiOutputClassifier也不行
def trainWithMultiOutputClassifier():
    from xgboost.sklearn import XGBClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from lightgbm.sklearn import LGBMClassifier

    dict, y_one_lable = encodingLable()
    y = y_one_lable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    rf = LGBMClassifier()
    rf = MultiOutputClassifier(rf)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('acc', acc)
    print('===========y_test[10]', y_test[10])
    print('===========y_pred[10]', y_pred[10])
    print('===========y_pred2', rf.predict([X_test[10]]))


if __name__ == '__main__':
    # lightgbm_learn()
    # encodingLable()
    trainWithMultiOutputClassifier()
    # random_forest()
    # cross_val_score_()