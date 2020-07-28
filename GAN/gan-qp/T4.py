#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/4/27 12:44                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np


inputs = tf.random_normal([10,50,41,20],dtype=tf.float32)
result = layers.conv2d(inputs,30,kernel_size=(4,4),stride=2,padding='valid')
result1 = layers.conv2d(inputs,30,kernel_size=(4,4),stride=2,padding='same')
x2 = tf.Variable(np.random.randn(10,25,20,30),dtype=tf.float32)

deconv = layers.conv2d_transpose(result,1,kernel_size=(4,4),stride=2,padding='valid')
deconv1 = layers.conv2d_transpose(result1,1,kernel_size=(4,4),stride=2,padding='same')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(result)
    sess.run(result1)
    print(result.shape)#(10,24,19,30)
    print(result1.shape)#(10,25,21,30)
    sess.run(deconv)
    sess.run(deconv1)
    print(deconv.shape)#(10, 50, 40, 1)
    print(deconv1.shape)#(10, 50, 42, 1)