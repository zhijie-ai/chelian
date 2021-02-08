#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/8/28 15:16
 =================知行合一=============
 HMM在股票市场中的应用
 我们假设隐藏状态数量是6，即假设股市的状态有6种，虽然我们并不知道每种状态到底是什么，
 但是通过后面的图我们可以看出那种状态下市场是上涨的，哪种是震荡的，哪种是下跌的。
 可观测的特征状态我们选择了3个指标进行标示，进行预测的时候假设假设所有的特征向量的状态服从高斯分布
'''

from hmmlearn.hmm import GaussianHMM
import numpy as np
from matplotlib import cm,pyplot as plt
import matplotlib.dates
import pandas as pd
import datetime
from pandas_datareader import DataReader

# from sklearn.datasets import

# 测试时间从2005年1月1日到2015年12月31日，拿到每日沪深300的各种交易数据。

# beginDate = '2005-01-01'
# endDate ='2015-12-31'
beginDate = datetime.datetime(2015,1,1)
endDate =datetime.datetime(2015,12,31)
n = 6 # 6个隐藏状态
df = DataReader('XOM','yahoo',beginDate,endDate)
print(df.head())
df.columns = ['OpeningPx','HighPx','LowPx','ClosingPx','Adj Close','TotalVolumeTraded']
print(df.head())
