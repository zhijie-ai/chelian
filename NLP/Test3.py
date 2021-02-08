#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/7/8 23:57                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import keras.backend as K
import numpy as np

# 生成输入数据
x1 = tf.convert_to_tensor([[1,2,3],[4,5,6]])
print(x1.shape)
x2 = tf.convert_to_tensor([[1,2,3],[4,5,6]])

sess = tf.Session()

print(sess.run(K.batch_dot(x1,x2,axes=1)))#相当于矩阵乘法后按该axis相加
print(sess.run(K.batch_dot(x1,x2,axes=1)).shape)#(2, 1)

# axes为1 的batch_dot输出如下：
# array([[14],
# [77]], dtype=int32)

# axes为2的batch_dot输出如下：
print(K.batch_dot(x1,x2,axes=0).shape)
print(sess.run(K.batch_dot(x1,x2,axes=0)))#如果去掉,axes=0就会报错，可以推断出，加了axis参数，是按元素相乘
    # 在按轴reduce操作

# array([[17],
# [29],
# [45]], dtype=int32)

# 实际上与先经过位对位乘法然后按某一个轴作聚合加法返回的结果一直，下面是验证结果。
print(tf.reduce_sum(tf.multiply(x1 , x2) , axis=0).shape)
print(sess.run(tf.reduce_sum(tf.multiply(x1 , x2) , axis=0)))

# array([17, 29, 45], dtype=int32)

print(tf.reduce_sum(tf.multiply(x1 , x2) , axis=1).shape)

# array([14, 77], dtype=int32)