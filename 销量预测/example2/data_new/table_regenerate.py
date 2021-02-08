#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/12/8 22:10                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np
import pandas as pd

user_cols = ['USER_ID','SHOP_ID','TIME_STA']# 支付时间，精确到秒
user_pay = pd.read_csv('../../example1/data/user_pay.txt',header=None, names = user_cols)
user_view = pd.read_csv('../../example1/data/user_view.txt',header=None,names = user_cols)
#user_view_extra = pd.read_csv('../../sample1/data/extra_user_view.txt',header=None,names = user_cols)
#user_view = pd.concat([user_view, user_view_extra])

#%%

user_pay['TIME_STA'] = pd.to_datetime(user_pay['TIME_STA'])
user_pay['DATE'] = pd.to_datetime(user_pay['TIME_STA']).dt.date
user_pay['HOUR'] = pd.to_datetime(user_pay['TIME_STA']).dt.hour

#每个用户在每个商店每小时的购买次数统计及修正
user_pay_new = user_pay.groupby(by =['USER_ID','SHOP_ID','DATE','HOUR'],as_index = False).count()
user_pay_new = user_pay_new.rename(columns={'TIME_STA':'Num_raw'})#按小时统计支付次数
user_pay_new['Num_post'] = 1+ np.log(user_pay_new['Num_raw'])/ np.log(2)#采用公式修正次数
#按小时统计每个商店的支付次数
user_pay_new = user_pay_new.groupby(by =['SHOP_ID','DATE','HOUR'],as_index = False).sum()
user_pay_new['DofW'] = pd.to_datetime(user_pay_new['DATE']).dt.dayofweek
user_pay_new = user_pay_new.drop('USER_ID', 1)
#shop_id,date,hour,Num_post
user_pay_new.to_csv('user_pay_new.csv',index = False)

#%%
user_view['TIME_STA'] = pd.to_datetime(user_view['TIME_STA'])
user_view['DATE'] = pd.to_datetime(user_view['TIME_STA']).dt.date
user_view['HOUR'] = pd.to_datetime(user_view['TIME_STA']).dt.hour

user_view_new = user_view.groupby(by =['USER_ID','SHOP_ID','DATE','HOUR'],as_index = False).count()
user_view_new = user_view_new.rename(columns={'TIME_STA':'Num_raw'})
user_view_new['Num_post'] = 1+ np.log(user_view_new['Num_raw'])/ np.log(2)
user_view_new = user_view_new.groupby(by =['SHOP_ID','DATE','HOUR'],as_index = False).sum()
user_view_new['DofW'] = pd.to_datetime(user_view_new['DATE']).dt.dayofweek
user_view_new = user_view_new.drop('USER_ID', 1)
#shop_id,date,hour,Num_post
user_view_new.to_csv('user_view_new.csv',index = False)