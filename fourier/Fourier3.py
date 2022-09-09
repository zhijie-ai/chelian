#-*- encoding=utf-8 -*-
#__author__:'xiaojie'

#滤出图像中的低频信号

import numpy as np
from PIL import Image
from numpy.fft import fft,ifft

def filterImage(srcImage):
    srcIm = Image.open(srcImage)#打开图像文件并获取数据
    srcArray = np.fromstring(srcIm.tobytes(), dtype=np.int8)
    result = fft(srcArray)#傅里叶变换并滤除低频信号
    result = np.where(np.absolute(result) < 9e4, 0, result)
    result = ifft(result)#傅里叶反变换,保留实部
    result = np.int8(np.real(result))
    im = Image.frombytes(srcIm.mode, srcIm.size, result)
    im.show()

filterImage('../data/image.jpg')
