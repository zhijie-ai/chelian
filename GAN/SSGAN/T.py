#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/5/7 15:54                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import numpy as np

def my_func(array1,array2):
    return array1 + array2, array1 - array2

def main():
    a = tf.placeholder(tf.float32, shape=[1, 2], name="tensor_a")
    b = tf.placeholder(tf.float32, shape=[None, 2], name="tensor_b")
    print('AAAAAAAAAA',b.get_shape())
    tile_a = tf.tile(a, [b.get_shape()[0], 1])
    sess = tf.Session()
    array_a = np.array([[1., 2.]])
    array_b = np.array([[3., 4.], [5., 6.], [7., 8.]])
    feed_dict = {a: array_a, b: array_b}
    tile_a_value = sess.run(tile_a, feed_dict=feed_dict)
    print(tile_a_value)

def main2():
    array1 = np.array([[1, 2], [3, 4]])
    array2 = np.array([[1, 2], [3, 4]])
    a1 = tf.placeholder(tf.float32, [2, 2], name='array1')
    a2 = tf.placeholder(tf.float32, [2, 2], name='array2')
    y1, y2 = tf.py_func(my_func, [a1, a2], [tf.float32, tf.float32])
    y3 = a1 + a2
    y4 = a1 - a2

    with tf.Session() as sess:
        y1_, y2_ = sess.run([y1, y2], feed_dict={a1: array1, a2: array2})
        y3_, y4_ = sess.run([y3, y4], feed_dict={a1: array1, a2: array2})
        print(y1_)
        print('*' * 10)
        print(y2_)
        print('A' * 10)
        print(y3_)
        print('A' * 10)
        print(y4_)


if __name__ =='__main__':
    main()
