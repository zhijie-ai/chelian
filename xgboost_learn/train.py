#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/8/17 14:17
 =================知行合一=============
 使用xgboost测试多分类
'''

import numpy as np
import xgboost as xgb

# label need to be 0 to num_class-1
data = np.loadtxt('../data/dermatology.data',delimiter=','
                  ,converters={33: lambda x:int(x =='?'),34:lambda x:int(x)-1})
sz = data.shape
print('data.shape',sz)
train = data[:int(sz[0] *0.7),:]
test= data[int(sz[0] * 0.7):,:]

train_X = train[:,:33]
train_y = train[:,34]

test_X = test[:,:33]
test_y = test[:,34]

xg_train = xgb.DMatrix(train_X,label=train_y)
xg_test = xgb.DMatrix(test_X,label=test_y)

#setup parameters for xgboost
param = {}
# use softmax multi_class classification
param['objective'] = 'multi:softmax'# 输出的直接是类别
# scale weight of positive example
param['eta'] = 0.1
param['max_depth']=6
param['silent']=1
param['nthread']=4
param['num_class']=6

watchlist = [(xg_train,'train'),(xg_test,'RL')]
num_round = 5
bst = xgb.train(param,xg_train,num_round,watchlist)
#get prediction
pred = bst.predict(xg_test)
print('pred',pred)
error_rate = np.sum(pred != test_y) / test_y.shape[0]
print('Test error using softmax = {}'.format(error_rate))

# do the same thing again,but output probilities
param['objective'] = 'multi:softprob' # 输出的是每个类别对应的概率
bst = xgb.train(param,xg_train,num_round,watchlist)
# Note: this convention has been changed since xgboost-unity
# get prediction,this is in 1D array, need reshape to (ndata,nclass)
pred_prob = bst.predict(xg_test)
print('pred_prob',pred_prob)
pred_label = np.argmax(pred_prob,axis=1)
print('pred_label',pred_label)
error_rate = np.sum(pred_label != test_y) /test_y.shape[0]
print('Test error using softprob={}'.format(error_rate))
