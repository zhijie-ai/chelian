#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/3/4 16:57
        
'''

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('./data/dataset.csv',encoding='gbk')
y=dataset['故障码Indexer']
X = dataset.drop(['故障码Indexer'],axis=1)

# 将字符串类型转换为数值类型及标准归一化操作
X = pd.get_dummies(X)
standardScaler = StandardScaler()
X = standardScaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

x = tf.placeholder(dtype=tf.float32,shape=(None,49))
y_ = tf.placeholder(dtype=tf.int64,shape=(None))

W = tf.Variable(tf.zeros([49,304]))
b = tf.Variable(tf.zeros([304]))

y_hat=tf.nn.softmax(tf.matmul(x,W)+b)

# 训练
# 定义损失函数，交叉熵损失函数
# 对于多分类问题，通常使用交叉熵损失函数
# reduction_indices等价于axis，指明按照每行加，还是按照每列加
# cross_entropy =  tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat,labels=y_)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 初始化变量
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(100):
    sess.run(train_step,feed_dict={x:X_train,y_:y_train})

# 评估m
# tf.argmax()是一个从tensor中寻找最大值序号，tf.argmax()就是求各个预测的数字中概率最大的那一个。
correct_prediction = tf.equal(y_,tf.argmax(y_hat,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 测试0.00280439
print(accuracy.eval({x: X_test, y_: y_test}))
#
# 总结
# 1，定义算法公式，也就是神经网络forward时的计算
# 2，定义loss，选定优化器，并指定优化器优化loss
# 3，迭代地对数据进行训练
# 4，在测试集或验证集上对准确率进行评测
