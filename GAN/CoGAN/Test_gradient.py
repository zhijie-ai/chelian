#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/7/4 15:03                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import  tensorflow as tf
V1 = tf.Variable(50,dtype='float32')
V2 = tf.Variable(20,dtype='float32')
loss = 2*V1+3*V2
opt = tf.train.GradientDescentOptimizer(0.1)
grad1 = opt.compute_gradients(loss,[V1,V2])
grad2 = tf.gradients(loss,[V1,V2])
grad3 = opt.compute_gradients(loss,[V2])
grad4 = tf.gradients(loss,[V2])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(grad1))
print(sess.run(grad2))
print(sess.run(grad3))
print(sess.run(grad4))