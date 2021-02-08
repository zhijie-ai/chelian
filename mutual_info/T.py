#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/4/18 15:11                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

import tensorflow as tf
import numpy as np
rng = np.random.RandomState(0)

data = rng.randint(1.0,10.0,(2,3,4,5))
# data = tf.constant_initializer(data,dtype=np.int32)
data = tf.constant(data,dtype=np.float32)
init = tf.global_variables_initializer()
result = tf.reduce_sum(data,axis=[1,2])
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(result))
    print(sess.run(tf.reduce_mean(result)))
    print(sess.run(tf.reduce_mean(tf.reduce_sum(result,-1))))
    print(sess.run(tf.reduce_mean(data)))