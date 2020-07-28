#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/11/16 21:00                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np
from scipy.sparse import csr

ix=None
dic = {'user':list('ABCDA'),'itme':[10,1,2,3,2]}

if ix == None:
    ix = dict()  # 存的是每个特征的值+列名:次数

n=5
g=2
nz = n * g

col_ix = np.empty(nz, dtype=int)

i = 0
for k, lis in dic.items():
    for t in range(len(lis)):
        ix[str(lis[t]) + str(k)] = ix.get(str(lis[t]) + str(k), 0) + 1
        col_ix[i + t * g] = ix[str(lis[t]) + str(k)]

    i += 1
p=None
row_ix = np.repeat(np.arange(0, n), g)
data = np.ones(nz)
if p == None:
    p = len(ix)

ixx = np.where(col_ix < p)

csr = csr.csr_matrix((data[ixx],(row_ix[ixx],col_ix[ixx])),shape=(n,p))
print(csr.toarray())

print(ix)
print(col_ix)
print(row_ix)
print(data)
print(ixx)

