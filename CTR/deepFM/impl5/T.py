#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/11/28 16:27                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import numpy as np

data = np.arange(25).reshape(5,5)
print(data)
W = tf.constant(data)
res = tf.nn.embedding_lookup(W,[0,1,0])
res2 = tf.nn.embedding_lookup(W, [[0],[1],[0]])
res4 = tf.nn.embedding_lookup(W, [[0,0,0,0],[1,1,1,1],[0,0,0,0]])
with tf.Session() as sess:
    res, res2, res4 = sess.run([res, res2, res4])

print(res)
print('aaaaaaaaaaaa')
print(res2)
print('aaaaaaaaaaaa')
print(res4)