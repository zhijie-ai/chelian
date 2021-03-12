#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/27 22:00                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np
import pandas as pd
import pickle
import logging
from collections import Counter

# for site_id, site_domain, app_id, app_domain, device_model,
# device_ip, device_id fields,C14,C17,C19,C21, one-hot using frequency
# for other fields, one-hot-encoding directly

# one-hot encoding directly
click = set()
hour = set()
C1 = set()
banner_pos = set()
site_category = set()
app_category = set()
device_type = set()
device_conn_type = set()
C15 = set()
C16 = set()
C18 = set()
C20 = set()

hours = set(range(24))

# one-hot encoding by frequency bucket
C14 = []
C17 = []
C19 = []
C21 = []
site_id = []
site_domain = []
app_id = []
app_domain = []
device_model = []
device_ip = []
device_id = []



train = pd.read_csv('data/ctr_data.csv',chunksize=4000)


for data in train:

    click_v = set(data['click'].values)
    click = click | click_v

    C1_v = set(data['C1'].values)
    C1 = C1 | C1_v

    C15_v = set(data['C15'].values)
    C15 = C15 | C15_v

    C16_v = set(data['C16'].values)
    C16 = C16 | C16_v

    C18_v = set(data['C18'].values)
    C18 = C18 | C18_v

    C20_v = set(data['C20'].values)
    C20 = C20 | C20_v

    banner_pos_v = set(data['banner_pos'].values)
    banner_pos = banner_pos | banner_pos_v

    site_category_v = set(data['site_category'].values)
    site_category = site_category | site_category_v

    app_category_v = set(data['app_category'].values)
    app_category = app_category | app_category_v

    device_type_v = set(data['device_type'].values)
    device_type = device_type | device_type_v

    device_conn_type_v = set(data['device_conn_type'].values)
    device_conn_type = device_conn_type | device_conn_type_v



# save dictionaries
with open('sets/click.pkl','wb') as f:
    pickle.dump(click,f)

with open('sets/hour.pkl','wb') as f:
    pickle.dump(hours,f)

with open('sets/C1.pkl','wb') as f:
    pickle.dump(C1,f)

with open('sets/C15.pkl','wb') as f:
    pickle.dump(C15,f)#C15代表了一个set，该field对应的值有几种类型，常规的one-hot编码类型

with open('sets/C16.pkl','wb') as f:
    pickle.dump(C16,f)

with open('sets/C18.pkl','wb') as f:
    pickle.dump(C18,f)

with open('sets/C20.pkl','wb') as f:
    pickle.dump(C20,f)

with open('sets/banner_pos.pkl','wb') as f:
    pickle.dump(banner_pos,f)

with open('sets/site_category.pkl','wb') as f:
    pickle.dump(site_category,f)

with open('sets/app_category.pkl','wb') as f:
    pickle.dump(app_category,f)

with open('sets/device_type.pkl','wb') as f:
    pickle.dump(device_type,f)

with open('sets/device_conn_type.pkl','wb') as f:
    pickle.dump(device_conn_type,f)

print(banner_pos)

