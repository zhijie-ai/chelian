#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/5/7 18:04                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

# Semi-supervised Learning GAN in Tensorflow
# 代码参考:https://github.com/gitlimlab/SSGAN-Tensorflow
# 本代码的实现相当于dataset中全是已经labeled的数据，并没有unlabeled的数据。
# 没有体现半监督学习的特性(只有少量的带标签的数据)