#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/30 16:56                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

# 用的次数当做值来进行one-hot

from itertools import count
from collections import defaultdict
from scipy.sparse import csr
import numpy as np
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm

def vectorize_dic(dic,ix=None,p=None,n=0,g=0):
    """
    dic -- dictionary of feature lists. Keys are the name of features
    ix -- index generator (default None)
    p -- dimension of feature space (number of columns in the sparse matrix) (default None)
    """
    if ix == None:
        ix = dict()# 记录了每个user_id和item_id出现的次数

    nz = n*g

    col_ix = np.empty(nz,dtype=int)

    i = 0
    for k,lis in dic.items():
        for t in range(len(lis)):
            ix[str(lis[t])+str(k)] = ix.get(str(lis[t])+str(k),0)+1
            col_ix[i+t*g] = ix[str(lis[t])+str(k)]

        i+=1
    # id = np.arange(0,10,2)
    # print(col_ix[id])
    # print(col_ix[(id+1)])
    print('ix',ix)
    print('col_ix',col_ix)

    row_ix  = np.repeat(np.arange(0,n),g)
    data = np.ones(nz)
    if p == None:
        p = len(ix)

    ixx = np.where(col_ix <p)
    print('ixx',ixx)
    print('data',data)
    print('data[ixx]',data[ixx])
    print('row_ix',row_ix)
    print('row_ix[ixx]',row_ix[ixx])
    print('col_ix',col_ix)
    print('col_ix[ixx]',col_ix[ixx])
    return csr.csr_matrix((data[ixx],(row_ix[ixx],col_ix[ixx])),shape=(n,p)),ix

cols = ['user','item','rating','timestamp']

train = pd.read_csv('data/ua2.base',delimiter='\t',names = cols)
test = pd.read_csv('data/ua.test',delimiter='\t',names = cols)

x_train,ix = vectorize_dic({'users':train['user'].values},n=len(train.index),g=1)
print('AAAAAA')
# https://blog.csdn.net/qq_33466771/article/details/80304498
print(x_train.toarray())
print(x_train.shape)




