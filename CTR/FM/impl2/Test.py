#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/11/25 13:23                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import pickle

print(pickle.load(open('field2count/C14.pkl','rb')))
print(pickle.load(open('sets/C1.pkl','rb')))
print(pickle.load(open('dicts/hour.pkl','rb')))
print(pickle.load(open('dicts/C1.pkl','rb')))
print(pickle.load(open('data/train_sparse_data_frac_0.01.pkl','rb')))