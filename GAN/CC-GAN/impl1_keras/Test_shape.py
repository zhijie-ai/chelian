#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/7/3 16:47                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf

inputs = tf.Variable(tf.random_normal([10,51,25,3]))
filter = tf.Variable(tf.random_normal([3,3,3,10]))

result = tf.nn.conv2d(inputs,filter,strides=[1,2,2,1],padding='SAME')
result2 = tf.nn.conv2d(inputs,filter,strides=[1,2,2,1],padding='VALID')
result3 = tf.nn.max_pool(inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
result4 = tf.nn.max_pool(inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(result).shape)#(10, 26, 26, 10)
print(sess.run(result2).shape)#(10, 25, 25, 10)
print(sess.run(result3).shape)#(10, 26, 13, 3)
print(sess.run(result4).shape)#(10, 25, 12, 3)
sess.close()