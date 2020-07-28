#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/4/28 10:57                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope

def generator(tensor):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0
    print (tensor.get_shape())
    with variable_scope.variable_scope('generator', reuse = reuse):
        tensor = slim.fully_connected(tensor, 1024)
        print( tensor)
        tensor = slim.batch_norm(tensor, activation_fn=tf.nn.relu)
        tensor = slim.fully_connected(tensor, 7*7*128)
        tensor = slim.batch_norm(tensor, activation_fn=tf.nn.relu)
        tensor = tf.reshape(tensor, [-1, 7, 7, 128])
        # print '22',tensor.get_shape()
        tensor = slim.conv2d_transpose(tensor, 64, kernel_size=[4,4], stride=2, activation_fn = None)
        print ('gen',tensor.get_shape())
        tensor = slim.batch_norm(tensor, activation_fn = tf.nn.relu)
        tensor = slim.conv2d_transpose(tensor, 1, kernel_size=[4, 4], stride=2, activation_fn=tf.nn.sigmoid)
        return tensor