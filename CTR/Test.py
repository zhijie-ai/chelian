#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'庄周梦蝶'                        #
# CreateTime:                                 #
#       2019/9/10 17:50                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def gen_data(size = 1000000):
    """生成数据
    输入数据X：在时间t，Xt的值有50%的概率为1，50%的概率为0；
    输出数据Y：在实践t，Yt的值有50%的概率为1，50%的概率为0，除此之外，如果`Xt-3 == 1`，Yt为1的概率增加50%， 如果`Xt-8 == 1`，则Yt为1的概率减少25%， 如果上述两个条件同时满足，则Yt为1的概率为75%。
    """
    X = np.array(np.random.choice(2,size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X,np.array(Y)

if __name__ == '__main__':
    X,y = gen_data(10)
    print(X)
    print(y)