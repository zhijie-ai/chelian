#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/12/4 10:02
 =================知行合一=============
'''

import numpy as np
import matplotlib.pyplot as plt
import math

k = np.arange(1,101)
mu = 4
def possion(k):
    f = (mu**k/math.factorial(k))*math.exp(-mu)
    return f

y = [possion(k1) for k1 in k]
plt.plot(k,y)
plt.show()