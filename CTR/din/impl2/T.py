#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/4/23 22:45                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import numpy as np

data = np.arange(24).reshape((2,3,4))
data = data.astype('float')
print(data)
data = tf.convert_to_tensor(data)
mean,variance=tf.nn.moments(data,[1,2])
print(data)
with tf.Session() as sess:
    print(sess.run(mean))