#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/18 20:34                       #
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
print(z2.shape)#(10, 12, 4, 10, 12, 8)
print(z.shape) #(10, 12, 4, 8)

print('*'*10)
w = tf.Variable(np.random.randint(10,size=(10,12,4,5)),dtype=tf.float32)
k = tf.Variable(np.random.randint(10,size=(10,12,5,8)),dtype=tf.float32)
z = tf.matmul(w,k)
print(z.shape) #(10, 12, 4, 8)

# print('='*10)
# a = K.ones((3,4,5,2))
# b = K.ones((2,5,3,7))
# c = K.dot(a, b)# 报错,将b中的3改为2，用dot就不会报错，就算改为2，用batch_dot还是会报错，可以推断出，batch_dot
#     # 应该前面要相同
# print(c.shape)
