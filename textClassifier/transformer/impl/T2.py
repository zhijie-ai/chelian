#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/22 15:17                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as tf

keys = tf.random.randn(2,3,16)

# Key Masking
key_masks = tf.sign(tf.sum(tf.abs(keys), axis=-1))  # (N, T_k)
key_masks = tf.tile(key_masks, [8, 1])  # (h*N, T_k)
key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(keys)[1], 1])  # (h*N, T_q, T_k)
print(key_masks)
