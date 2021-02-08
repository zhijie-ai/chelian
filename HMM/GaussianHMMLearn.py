#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/8/28 17:59
 =================知行合一=============
 观测值为连续值
'''

import numpy as np
from hmmlearn import hmm
import warnings

warnings.filterwarnings('ignore')

startprob = np.array([0.5,0.3,0.1,0.1])
# the transition matrix,note that theres are no transitions possible
# between component 1 and 3
transmat = np.array([[0.6,0.2,0.1,0.1],
                     [0.3,0.4,0.2,0.1],
                     [0.1,0.3,0.4,0.2],
                     [0.2,0.1,0.2,0.5]])
# the means of each component
means = np.array([[0.0,0.0],
                  [0.0,11.0],
                  [9.0,10.0],
                  [11.0,-1.0]])
# the convariance of each component
covars = .5*np.tile(np.identity(2),(4,1,1))
# build an hmm instance and set parameters
model = hmm.GaussianHMM(n_components=4,covariance_type='full')

# instead of fitting it from the data,we directly set the estimated
# parameters ,the means and covariance of the components
model.startprob_ = startprob
model.transmat_ = transmat
model.means_ = means
model.covars_= covars

print('A'*20)
sample,labels = model.sample(4)
print(sample)
print(labels)

# 现在我们跑一跑解码的过程，由于观测状态是二维的，我们用三维观测序列，所以这里的输入是一个3*2的张量
seen = np.array([[1.1,2.0],[-1,2.0],[3,7]])
logprob,state = model.decode(seen,algorithm='viterbi')
print(state)
#对数概率计算问题
print(model.score(seen))

