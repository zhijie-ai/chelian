# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/8/27 11:20                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
# 积分运算相关
import numpy as np
import scipy.integrate as si

T = 2*np.pi
t = np.linspace(0, 2*np.pi, 1000)

def f(x):
    return np.sin(5*x)*np.sin(5*x)
print(si.quad(f, 1, T)[0])