#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/4/30 13:50                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# https://blog.csdn.net/github_39611196/article/details/79184040
import tensorflow as tf
import cv2
import numpy as np

# 读取图片
img = cv2.imread("2.jpg")
# 显示原始图片
cv2.imshow("resource", img)
h, w, depth = img.shape
img = np.expand_dims(img, 0)
# 临界点插值
nn_image = tf.image.resize_nearest_neighbor(img, size=[h+100, w+100])
nn_image = tf.squeeze(nn_image)
with tf.Session() as sess:
    # 运行 'init' op
    nn_image = sess.run(nn_image)
nn_image = np.uint8(nn_image)

# 双线性插值
bi_image = tf.image.resize_bilinear(img, size=[h+100, w+100])
bi_image = tf.squeeze(bi_image)
with tf.Session() as sess:
    # 运行 'init' op
    bi_image = sess.run(bi_image)
bi_image = np.uint8(bi_image)

# 双立方插值算法
bic_image = tf.image.resize_bicubic(img, size=[h+100, w+100])
bic_image = tf.squeeze(bic_image)
with tf.Session() as sess:
    # 运行 'init' op
    bic_image = sess.run(bic_image)
bic_image = np.uint8(bic_image)

print(img.shape,nn_image.shape)
# 显示结果图片
cv2.imshow("result_nn", nn_image)
cv2.imshow("result_bi", bi_image)
cv2.imshow("result_bic", bic_image)
cv2.waitKey(0)
cv2.destroyAllWindows()