#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/3/12 13:31                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import pickle
import pandas as pd


def to_df(file_path):
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:
            df[i] = eval(line)
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df


reviews_df = to_df('../rawdata/reviews_Electronics_5.json')
with open('data/reviews.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

meta_df = to_df('../data/meta_Electronics.json')
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
meta_df = meta_df.reset_index(drop=True)
with open('../data/meta.pkl', 'wb') as f:
    pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)