#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/4/10 17:35                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import pickle

# with open('../raw_data/reviews.pkl', 'rb') as f:
#   reviews_df = pickle.load(f)
#   print(reviews_df.columns)

with open('../raw_data/meta.pkl', 'rb') as f:
  meta_df = pickle.load(f)
  print(meta_df.columns)
  meta_df = meta_df[['asin', 'categories']]
  print(meta_df.head())
  print(meta_df['categories'][0])
  # meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])
  # print(meta_df.head())
  # print(meta_df['categories'].head())
