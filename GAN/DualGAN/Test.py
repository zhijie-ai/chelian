#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/7/6 22:30                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
from keras.datasets import mnist
import scipy
import numpy as np
import matplotlib.pyplot as plt


(X_train,_),(_,_) = mnist.load_data()
print(X_train.shape)
X_train=X_train[0:10]
print(X_train.shape)
print(X_train[0].shape)
X_train = (X_train.astype(np.float32)-127.5)/127.5
X_train_ = scipy.ndimage.interpolation.rotate(X_train,90,axes=(1,2))
print(X_train_.shape)

# plt.subplot(1,2,1)
# plt.imshow(X_train_[0],cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(X_train[0],cmap='gray')
# plt.show()

fig ,ax = plt.subplots(2,2)
ax[0,1].imshow(X_train[0],cmap='gray')
ax[0,0].imshow(X_train_[0],cmap='gray')
plt.show()
print(ax)
