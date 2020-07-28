#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/11/26 14:15                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np

def transfer_data(sample,fields_dict,array_length):
    array = np.zeros([array_length])
    for field in fields_dict:
        # get index of array
        if field =='click':
            field_value = sample[field]
            ind = fields_dict[field][field_value]
            if ind ==(array_length-1):
                array[ind]=1
            else:
                array[ind]=-1
        else:
            field_value = sample[field]
            ind = fields_dict[field][field_value]
            array[ind]=1
    return array

def get_batch(x, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < x.shape[0] else x.shape[0]
    return x.iloc[start:end, :]