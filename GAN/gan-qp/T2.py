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

d1 = rng.randint(1,10,(3,4))
d2 = rng.randint(1,10,(3,4))

# data = tf.constant_initializer(data,dtype=np.int32)
print(d1)
print(d2)
print(np.mean(d1))
d1 = tf.constant(d1,dtype=np.float32)
d2 = tf.constant(d2,dtype=np.float32)
init = tf.global_variables_initializer()
r1 = tf.reduce_mean(d1)
r2 = tf.reduce_mean(d2)
r3 = r1-r2
r4 = tf.reduce_mean(d1-d2)
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(r1))
    print(sess.run(r2))
    print(sess.run(r3))
    print(sess.run(r4))