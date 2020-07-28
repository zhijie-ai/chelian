#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/4/27 18:14                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf

tf.contrib.layers.l2_regularizer()
with tf.variable_scope('name_scope_x'):
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
    with tf.variable_scope('ddd'):
        v2 = tf.get_variable(name='v2',shape=[2])
        v3 = tf.Variable(3)




with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name, sess.run(var1))
    print(v2.name, sess.run(v2))
    print(tf.trainable_variables())
    print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
