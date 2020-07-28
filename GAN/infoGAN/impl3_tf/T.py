#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/4/27 18:06                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf

def func(in_put, in_channel, out_channel):
    with tf.variable_scope(name_or_scope='', reuse=False):    ### 改动部分 ###
        weights = tf.get_variable(name="weights", shape=[2, 2, in_channel, out_channel],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        convolution = tf.nn.conv2d(input=in_put, filter=weights, strides=[1, 1, 1, 1], padding="SAME")
    return convolution


def main():
    with tf.Graph().as_default():
        input_x = tf.placeholder(dtype=tf.float32, shape=[1, 4, 4, 1])
        output = func(input_x, 1, 1)

        for _ in range(2):
            # output = func(input_x, 1, 1)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                import numpy as np
                _output = sess.run([output], feed_dict={input_x:np.random.uniform(low=0, high=255, size=[1, 4, 4, 1])})
                print(output)

if __name__ == "__main__":
    main()