#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/7/8 23:49                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import keras.backend as K
import tensorflow as tf
import numpy as np

w = K.variable(np.random.randint(10,size=(10,12,4,5)))
k = K.variable(np.random.randint(10,size=(10,12,5,8)))
z = K.batch_dot(w,k)
z2 = K.dot(w,k)
print(z.shape) #(10, 12, 4, 8)
print(z2.shape) #(10, 12, 4, 10, 12, 8)
print('=====================')
X = K.variable(np.random.randn(100,20))
y=K.variable(np.random.randn(100,30,20))

z3 = K.batch_dot(X,y,axes=[1,2])
print(z3.shape)#(100, 30)