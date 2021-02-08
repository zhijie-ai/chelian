#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/17 15:02                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import numpy as np
import warnings

warnings.filterwarnings('ignore')

data = np.random.randint(1,3,size=(2,3))
print(data)
data = tf.constant(data,dtype=tf.float32)

with tf.Session() as sess:
    print(sess.run(tf.nn.softmax(data))) # softmax的计算单位是行向量
    print(sess.run(tf.nn.softmax(data)).sum(axis=1))
    print(sess.run(tf.random_uniform([10], -0.05, 0.05)))
