#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/8/16 14:52
 =================知行合一=============
'''
# 使用多层感知机分类
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('./data/dataset.csv',encoding='gbk')
y=dataset['故障码Indexer']
X = dataset.drop(['故障码Indexer'],axis=1)

# 将字符串类型转换为数值类型
X = pd.get_dummies(X)
standardScaler = StandardScaler()
X = standardScaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

clf = MLPClassifier(solver='sgd',alpha=1e-5,
                    hidden_layer_sizes=(30,50),max_iter=1000,tol=1e-3)
clf.fit(X,y)
clf.partial_fit

y_pred=clf.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print('acc:',acc)

pprint.pprint([coef.shape for coef in clf.coefs_])

