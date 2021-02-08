#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/20 19:05                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import numpy as np

data = np.random.randn(10,80,80,20)
print(data.shape)
data = tf.convert_to_tensor(data,dtype=tf.float32)
mean, variance = tf.nn.moments(data, list(range(len(data.get_shape())-1)))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(mean).shape)
    print(sess.run(variance).shape)