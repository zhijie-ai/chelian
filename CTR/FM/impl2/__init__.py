#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'庄周梦蝶'                        #
# CreateTime:                                 #
#       2019/9/1 20:37                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# https://github.com/zhijie-ai/CTR_Prediction/tree/master/FM

# 1.将数据字段分成直接one-hot编码和出现的频率分桶之后再one-hot编码2种类型
# 2.直接one-hot类型的field存到sets目录下，比如C15存的是该field下值的set集合step1.py，文件
# 需要分桶的field存的是key对应到某个桶的次数，在field2count目录下,Data_analysis.py文件
# 3.在step2中，把所有的key都打一个index，作为将来数据的维度位置step2.py文件