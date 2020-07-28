#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/7/5 21:23                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
from scipy import misc
import matplotlib.pyplot as plt

img = misc.imread('./facades/train/1.jpg')
img = img/127.5 - 1.
img = img*0.5+0.5# 如果不将数据变成0-1之间，显示会报错，像素值不能为负数
plt.imshow(img)
# plt.show()
plt.savefig('t.png')
print(img.max(),img.min())