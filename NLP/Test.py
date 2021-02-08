#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/7/8 23:43                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# dot比batch_dot更智能
from keras import backend as K
a = K.ones((2,3,4))
b = K.ones((7,4,5))
c = K.dot(a, b)
print(c.shape)#(2, 3, 7, 5)
print('*'*10)
from keras import backend as K
a = K.ones((1, 2, 4))
b = K.ones((8, 7, 4, 5))
c = K.dot(a, b)
print(c.shape)# (1, 2, 8, 7, 5).
print('*'*10)
from keras import backend as K
a = K.ones((3, 7, 4, 2))
b = K.ones((9, 8, 7, 2, 5))
c = K.dot(a, b)
print(c.shape) #(9, 8, 7, 4, 5)
print('aaaaaaaaaa')
# from keras import backend as K
# a = K.ones((3, 7, 4, 2))
# b = K.ones((9, 8, 7, 2, 5))
# c = K.batch_dot(a, b)
# print(c.shape) #会报错
