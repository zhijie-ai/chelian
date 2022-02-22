# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2021/10/19 21:00                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

labels = tf.convert_to_tensor([[0.2, 0.3, 0.5],
          [0.1, 0.6, 0.3]])
logits = tf.convert_to_tensor([[2, 0.5, 1],
          [0.1, 1,3]])
res1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
res2 = K.categorical_crossentropy(labels, logits, from_logits=True)

labels1 = tf.convert_to_tensor([0,2])
logits1 = tf.convert_to_tensor([[2, 0.5, 1],[0.1, 1, 3]])
res3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels1, logits=logits1)
res4 = K.sparse_categorical_crossentropy(labels1, logits1, from_logits=True)
with tf.Session() as sess:
    print(sess.run(res1))
    print(sess.run(res2))
    print(sess.run(res3))
    print(sess.run(res4))


