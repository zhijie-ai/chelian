#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/6/26 15:03                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np
import tensorflow as tf
from keras.metrics import binary_crossentropy

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

# 5个样本三分类问题，且一个样本可以同时拥有多类
y=np.array([0.9,0.8,1.0,1.0,1.0,1.0,0.2])
logits = np.array([0.2,0.5,0.8,12.0,15.0,18.0,20.0])
# 按计算公式计算
y_pred = sigmoid(logits)
E1 = -y*np.log(y_pred)-(1-y)*np.log(1-y_pred)
print(E1.mean())     # 按计算公式计算的结果
# 按封装方法计算
sess =tf.Session()
y = np.array(y).astype(np.float64) # labels是float64的数据类型
E2 = sess.run(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits))
print(E2.mean())     # 按 tf 封装方法计算

y = tf.convert_to_tensor(y)
y_pred = tf.convert_to_tensor(y_pred)
y_pred2 = tf.convert_to_tensor(logits)
E3 = binary_crossentropy(y_true=y,y_pred=y_pred)#经过了sigmoid函数激活过的
print(sess.run(E3))
print(sess.run(E3*y))
print(sess.run(tf.reduce_mean(E3*y)))

if E1.all() == E2.all():
    print("True")
else:
    print("False")

