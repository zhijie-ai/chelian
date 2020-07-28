#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/5/19 16:56                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import  tensorflow as tf
import numpy as np

data = np.random.randint(1,5,size=(2,3,4))
# print(np.mean(data[0],axis=1),np.mean(data[0],axis=2))
print(np.sum(data[0]))
print(np.sum(data))
print(data)
data = tf.constant(data)
op1 = tf.reduce_sum(data,axis=[1,2])# 将这2个维度的数据加起来
op2 = tf.reduce_sum(data)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(sess.run(op1))
print(sess.run(op2))
sess.close()