#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/3/29 14:53                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
# 生成训练集的时候，都是用隐式反馈的正例，然后给每条数据随机生成n条负例,其实可以自己控制，降低热门item成为正样本的概率，
#     提升热门item成为负样本的概率