#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/19 21:10                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import numpy as np

data = np.random.randn(128,200,512)
data2 = np.random.randn(6,8,512)

print(data.shape)
data = tf.convert_to_tensor(data)

position_enc = np.array([
                [pos / np.power(10000, (i - i % 2) / 10) for i in range(10)]
                for pos in range(8)])
print(position_enc.shape)

data = tf.layers.dense(data,512,use_bias=False)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(data).shape)#(128, 200, 512)
    print(sess.run(tf.reduce_sum(data,axis=-1)).shape)#(128, 200)
    print('A'*10)
    print(sess.run(tf.sign(tf.reduce_sum(tf.abs(data2), axis=-1))))
    print(sess.run(tf.expand_dims(tf.range(6.5), 0)))