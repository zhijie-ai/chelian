#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/22 17:16                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

import tensorflow as tf

inputs = [[5,6,7,0,0]]

lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[10, 6],
                                       initializer=tf.contrib.layers.xavier_initializer())
lookup_table = tf.concat((tf.zeros(shape=[1, 6]),
                                      lookup_table[1:, :]), 0)
outputs = tf.nn.embedding_lookup(lookup_table, inputs)
key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(outputs), axis=-1)), -1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(outputs))
    print(sess.run(outputs).shape)
    print(sess.run(outputs)[0,-1:,].sum())
    print(sess.run(outputs)[0,1,].sum())
    print('A'*10)
    print(sess.run(key_masks))