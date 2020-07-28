#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/5/21 14:26                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import numpy as np

tensor = tf.constant([[1, 2, 3],[1, 2, 3]])
print(tensor)
paddings = tf.constant([[1, 2], [3, 4]])
result = tf.pad(tensor, paddings)

rng = np.random.RandomState(10)
d1 = rng.randint(10,size=(2,3,4))
d2 = rng.randint(10,size=(2,3,4))
print(d1)
print(d2)

d3 = tf.convert_to_tensor(d1)
d4 = tf.convert_to_tensor(d2)

res = tf.matmul(d3,d4,transpose_b=True)

with tf.Session() as sess:
    print(sess.run(res))

print('AAAAAAAA',np.dot(d1[0],d2[0].T))