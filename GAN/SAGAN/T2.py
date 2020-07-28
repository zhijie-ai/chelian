#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/5/22 11:10                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf

t = tf.constant([[[1, 2, 3], [2, 3, 4], [2, 1, 4]],
                 [[1, 2, 3], [2, 3, 4], [2, 1, 4]],
                 [[1, 2, 3], [2, 3, 4], [2, 1, 4]]])
t2 = tf.constant([[1, 2, 3]])
print(t.get_shape())
print(t2.get_shape())

a = tf.pad(t, [[1, 1], [2, 2], [3, 3]])
c = tf.pad(t2, [[1, 1], [2, 2]])

d = tf.constant([[1, 2, 3], [2, 3, 4], [2, 1, 4]])
print('d.shape',d.shape)
e = tf.pad(d, [[1, 2], [3, 4]])#可以从增加维度的角度来理解。每个list代表各个维度

f = tf.constant([1,2,4])
g = tf.pad(f,[[1,2]])

with tf.Session() as sess:
    # a, c = sess.run([a, c])
    # print(a.shape)
    # print(c.shape)
    # print('*'*10)
    print('=======================')
    print(sess.run(e).shape)
    print(sess.run(e))
    print('++++++++++++++++++')
    print(sess.run(g).shape)
    print(sess.run(g))
