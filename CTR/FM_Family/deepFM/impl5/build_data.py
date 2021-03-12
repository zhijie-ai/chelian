#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/11/27 20:57                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import pandas as pd


def load_data():
    train_data = {}

    file_path = 'data/tiny_train_input2.csv'
    data = pd.read_csv(file_path, header=None)
    data.columns = ['c' + str(i) for i in range(data.shape[1])]
    label = data.c0.values
    label = label.reshape(len(label), 1)
    train_data['y_train'] = label

    co_feature = pd.DataFrame()#连续特征
    ca_feature = pd.DataFrame()#离散特征
    ca_col = []
    co_col = []#列名
    feat_dict = {}
    cnt = 1
    for i in range(1, data.shape[1]):
        target = data.iloc[:, i]
        col = target.name
        l = len(set(target))
        if l > 10:# >10,认为是连续特征
            target = (target - target.mean()) / target.std()#标准化
            co_feature = pd.concat([co_feature, target], axis=1)
            feat_dict[col] = cnt
            cnt += 1
            print(cnt)
            co_col.append(col)
        else:#认为是离散特征,每个类别都是一个维度，相当于one-hot
            us = target.unique()
            feat_dict[col] = dict(zip(us, range(cnt, len(us) + cnt)))
            ca_feature = pd.concat([ca_feature, target], axis=1)
            cnt += len(us)
            print(cnt)
            ca_col.append(col)
    feat_dim = cnt
    feature_value = pd.concat([co_feature, ca_feature], axis=1)# X
    feature_index = feature_value.copy()
    for i in feature_index.columns:
        if i in co_col:#age:10
            feature_index[i] = feat_dict[i]
        else:# 离散特征，特征值变为1.# gender:{'boy':12,'gril':13}
            feature_index[i] = feature_index[i].map(feat_dict[i])
            feature_value[i] = 1.

    train_data['xi'] = feature_index.values.tolist()
    train_data['xv'] = feature_value.values.tolist()
    train_data['feat_dim'] = feat_dim
    print(feature_index)
    print(feature_value)
    print(feat_dim)
    return train_data

if __name__ == '__main__':
   load_data()
   from gensim.models import Word2Vec
   Word2Vec()