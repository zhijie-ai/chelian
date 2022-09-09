#-*- encoding=utf-8 -*-
#__author__:'xiaojie'

#二级傅里叶变换

import numpy as np
import matplotlib.pyplot as plt
x = np.random.rand(10,10)
wave = np.cos(x)
# plt.plot(wave)


transformed = np.fft.fft2(wave)
# plt.plot(transformed)
#同样地二维反变换也使用ifft2()函数。
plt.plot(np.fft.ifft2(transformed))



plt.show()