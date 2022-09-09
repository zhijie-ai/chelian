#-*- encoding=utf-8 -*-
#__author__:'xiaojie'

'''
傅里叶变换
'''

import numpy as np
import  scipy
import matplotlib.pyplot as plt
x = np.linspace(0,2*np.pi,50)
wave = np.cos(x)
transformed = np.fft.fft(wave)#傅里叶变换
# transformed = scipy.fft(wave)
plt.plot(transformed)

# plt.plot(np.fft.ifft(transformed))#反变换

#移频(针对作好傅里叶变换的数据)
#使用np.fft中的ffshift可以对信号进行移频操作，还有iffshift可以将移频后的信号还原成之前的
shifted = np.fft.fftshift(transformed)#移频
# plt.plot(shifted)
# plt.plot(np.fft.ifftshift(shifted))

#*对移频后的信号进行傅里叶反变换
# plt.plot(np.fft.ifft(shifted))












plt.show()