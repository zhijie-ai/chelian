
#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/12/7 16:23
 =================知行合一=============
'''

import numpy as np
import math
import matplotlib.pyplot as plt


def gaussian(x, u, sigma):
    """
    param x:要计算概率密度值的点
    param u:均值
    param sigma:方差
    return x的概率密度值
    """
    return math.exp(-(x - u) ** 2 / (2 * sigma * sigma)) / math.sqrt(2 * math.pi * sigma * sigma)


def importance_sampling(ori_sigma, sample_sigma):
    """
    param ori_sigma:原始分布p(x)的方差
    param sample_sigma:采样分布p~(x)的方差
    return

    """
    origin = []
    for n in range(10):
        # 进行10次计算
        Sum = 0
        for i in range(100000):
            a = np.random.normal(1.0, ori_sigma)
            Sum += a
        origin.append(Sum)

    isample = []
    for n in range(10):
        Sum2 = 0
        for i in range(100000):
            a = np.random.normal(1.0, sample_sigma)  # 计算从正太分布采样出来的x
            ua = gaussian(a, 1.0, sample_sigma)  # 计算采样概率密度
            na = gaussian(a, 1.0, ori_sigma)  # 计算原始概率密度
            Sum2 += a * na / ua
        isample.append(Sum2)

    origin = np.array(origin)
    isample = np.array(isample)

    print(np.mean(origin), np.std(origin))
    print(np.mean(isample), np.std(isample))


importance_sampling(1.0, 1.0)
importance_sampling(1.0, 0.5)
importance_sampling(1.0, 2.0)

xs = np.linspace(-5, 6, 301)
y1 = [gaussian(x, 1.0, 1.0) for x in xs]
y2 = [gaussian(x, 1.0, 0.5) for x in xs]
y3 = [gaussian(x, 1.0, 2.0) for x in xs]

fig = plt.figure(figsize=(8, 5))

plt.plot(xs, y1, label="sigma=1.0")
plt.plot(xs, y2, label="sigma=0.5")
plt.plot(xs, y3, label="sigma=2.0")
plt.legend()
plt.show()

