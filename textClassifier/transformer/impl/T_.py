#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/22 14:41                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import numpy as np

images = np.random.random([5,2])
label = np.asarray(range(0, 5))
images = tf.cast(images, tf.float32)
label = tf.cast(label, tf.int32)
input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
# 将队列中数据打乱后再读取出来
image_batch, label_batch = tf.train.shuffle_batch(input_queue, batch_size=10, num_threads=1, capacity=64, min_after_dequeue=1)

sv = tf.train.Supervisor()
with sv.managed_session() as sess:
    image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
    for j in range(10):
        # print(image_batch_v.shape, label_batch_v[j])
        print(image_batch_v[j]),
        print(label_batch_v[j])