#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/18 20:48                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# 验证K.batch_dot方法，2个输入维度相同的操作
# 可以推断出，K.batch_dot方法，如果2个参数的shape相同是按元素相乘的操作，如果不同，按最后2维做矩阵乘法
# 因此，最后2维应该满足矩阵乘法的条件,

import keras.backend as K
import numpy as np
import tensorflow as tf

# w = tf.convert_to_tensor([[1,2,3],[4,5,6]])
# k = tf.convert_to_tensor([[1,2,3],[4,5,6]])
# z = K.batch_dot(w,k)# 输入的shape相同，如果不加axis参数会报错

# w = K.variable(np.random.randint(10,size=(10,12,4,5)))
# k = K.variable(np.random.randint(10,size=(10,12,5,8)))
# z = K.batch_dot(w,k)
# z2 = K.batch_dot(w,k,axes=1)
# print(z.shape)
# print(z2.shape)#报错

w = K.variable(np.random.randint(10,size=(10,4,5,5)))
k = K.variable(np.random.randint(10,size=(10,4,5,1)))
z = K.batch_dot(w,k)
z2 = K.batch_dot(w,k)
print(z.shape)
print(z2.shape)
print(tf.reshape(w,tf.concat([tf.shape(w), [1] * (2)],axis=0)).shape)

# w = K.variable(np.random.randint(10,size=(100,20)))
# k = K.variable(np.random.randint(10,size=(100,30,20)))
# z2 = K.batch_dot(w,k,axes=(1,2))
# print(z2.shape)#(100, 30)

sess = tf.Session()
print(sess.run(tf.concat([tf.shape(w), [1] * (1)], axis=0)))#[10  4  5  5  1]

w = K.variable(np.random.randint(10,size=(100,1,20)))
k = K.variable(np.random.randint(10,size=(100,20,30)))
print(K.batch_dot(w,k).shape)
