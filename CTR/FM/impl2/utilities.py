#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/27 21:57                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import pandas as pd
import pickle
import logging
from scipy.sparse import coo_matrix
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

#sample为一条记录，issample为一条数据里面的第几列，fields_dict中，key为字段名，value为dict
def one_hot_representation(sample, fields_dict, isample):
    """
    One hot presentation for every sample data
    :param fields_dict: fields value to array index
    :param sample: sample data, type of pd.series
    :param isample: sample index
    :return: sample index
    """
    index = []
    for field in fields_dict:
        # get index of array
        if field == 'hour':
            field_value = int(str(sample[field])[-2:])
        else:
            field_value = sample[field]
        ind = fields_dict[field][field_value]
        index.append([isample,ind])#one-hot之前第几列的数据对应值的索引
    return index

# fields_dict是个dict，key为fields字段，value为一个dict
def train_sparse_data_generate(train_data, fields_dict):
    sparse_data = []
    # batch_index
    ibatch = 0
    for data in train_data:
        labels = []
        indexes = []
        for i in range(len(data)):
            sample = data.iloc[i,:]
            click = sample['click']
            # get labels
            if click == 0:
                label = 0
            else:
                label = 1
            labels.append(label)
            # get indexes
            index = one_hot_representation(sample,fields_dict, i)#(第几列,index)
            indexes.extend(index)
        sparse_data.append({'indexes':indexes, 'labels':labels})
        ibatch += 1
        if ibatch % 200 == 0:
            logging.info('{}-th batch has finished'.format(ibatch))
    with open('data/train_sparse_data_frac_0.01.pkl','wb') as f:
        pickle.dump(sparse_data, f)

def t1est_sparse_data_generate(test_data, fields_dict):
    sparse_data = []
    # batch_index
    ibatch = 0
    for data in test_data:
        ids = []
        indexes = []
        for i in range(len(data)):
            sample = data.iloc[i,:]
            ids.append(sample['id'])
            index = one_hot_representation(sample,fields_dict, i)
            indexes.extend(index)
        sparse_data.append({'indexes':indexes, 'id':ids})
        ibatch += 1
        if ibatch % 200 == 0:
            logging.info('{}-th batch has finished'.format(ibatch))
    with open('data/train_sparse_data_frac_0.01.pkl','wb') as f:
        pickle.dump(sparse_data, f)


# generate batch indexes
if __name__ == '__main__':

    fields = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
              'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
              'app_id', 'app_category', 'device_model', 'device_type', 'device_id',
              'device_conn_type']
    batch_size = 512
    train = pd.read_csv('data/ctr_data.csv', chunksize=batch_size)
    # loading dicts
    fields_dict = {}
    for field in fields:
        with open('dicts/'+field+'.pkl','rb') as f:#如果是连续型是dict，离散型为set
            fields_dict[field] = pickle.load(f)#key:index,即该特征下每个值对应的索引

    # test_sparse_data_generate(test, fields_dict)
    train_sparse_data_generate(train, fields_dict)
    print(fields_dict)