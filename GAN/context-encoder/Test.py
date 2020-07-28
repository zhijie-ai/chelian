#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/7/1 18:04                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import glob
from scipy import misc

height = 218
width = 178
mask_height = 64
mask_width = 64

def mask_randomly(imgs):
    y1 = np.random.randint(0, height - mask_height, imgs.shape[0])
    y2 = y1 + mask_height
    x1 = np.random.randint(0, width - mask_width, imgs.shape[0])
    x2 = x1 + mask_width

    masked_imgs = np.empty_like(imgs)
    missing_parts = np.empty((imgs.shape[0], mask_width, mask_width,3))
    for i, img in enumerate(imgs):
        masked_img = img.copy()
        _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]
        missing_parts[i] = masked_img[_y1:_y2, _x1:_x2, :].copy()
        masked_img[_y1:_y2, _x1:_x2, :] = 0
        masked_imgs[i] = masked_img

    return masked_imgs, missing_parts, (y1, y2, x1, x2)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# X_train = X_train[0:10,:,:,:]
# plt.imshow(X_test[100])
# plt.show()
# misc.imsave('3.png',X_train[0])


# imgs = glob.glob('../../images/img_align_celeba/*.jpg')
# X_train = np.array([misc.imread(f) for f in imgs[0:1]])
# misc.imsave('1.png',X_train[0])
# X,_ ,id= mask_randomly(X_train)
# print(id)
# plt.imshow(X_train[0])
# plt.show()
# plt.imshow(X[0])
# plt.show()
# misc.imsave('2.png',X[0])
print((y_train==3).flatten())
