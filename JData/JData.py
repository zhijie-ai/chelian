#!/usr/bin/env python
# coding: utf-8

# # 大赛简介

# # 数据介绍

# In[1]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from datetime import timedelta
from datetime import datetime

warnings.filterwarnings('ignore')


# In[2]:


# 定义文件名
ACTION_201602_FILE = "data/JData_Action_201602.csv"
ACTION_201603_FILE = "data/JData_Action_201603.csv"
ACTION_201604_FILE = "data/JData_Action_201604.csv"
COMMENT_FILE = "data/JData_Comment.csv"
PRODUCT_FILE = "data/JData_Product.csv"
USER_FILE = "data/JData_User.csv"
USER_TABLE_FILE = "data/User_table.csv"
ITEM_TABLE_FILE = "data/Item_table.csv"


# # 数据分析与预处理

# ## 异常值判断

# 根据官方给出的数据介绍里，可以知道数据可能存在哪些异常信息
# 
# * 用户文件
#     * 用户的age存在未知的情况，标记为-1
#     * 用户的sex存在保密情况，标记为2
#     * 后续分析发现，用户注册日期存在系统异常导致在预测日之后的情况，不过目前针对该特征没有想法，所以不作处理
# * 商品文件
#     * 属性a1,a2,a3均存在未知情形，标记为-1
# * 行为文件
#    * model_id为点击模块编号，针对用户的行为类型为6时，可能存在空值
# 

# ## 3.6空值判断

# In[3]:


def check_empty(file_path, file_name):
    df_file = pd.read_csv(file_path,nrows =500000)
    print('Is there any missing value in {0}? {1}'.format(file_name, df_file.isnull().any().any()) )
check_empty(USER_FILE, 'User')
check_empty(ACTION_201602_FILE, 'Action 2')
check_empty(ACTION_201603_FILE, 'Action 3')
check_empty(ACTION_201604_FILE, 'Action 4')
check_empty(COMMENT_FILE, 'Comment')
check_empty(PRODUCT_FILE, 'Product')


# 由上述简单的分析可知，用户表及行为表中均存在空值记录，而评论表和商品表则不存在，但是结合之前的数据背景分析商品表中存在属性未知的情况，后续也需要针对分析，进一步的我们看看用户表和行为表中的空值情况

# In[4]:


def empty_detail(f_path, f_name):
    df_file = pd.read_csv(f_path,nrows=500000)
    print('empty info in detail of {0}:'.format(f_name))
    print (pd.isnull(df_file).any())
empty_detail(USER_FILE, 'User')
empty_detail(ACTION_201602_FILE, 'Action 2')
empty_detail(ACTION_201603_FILE, 'Action 3')
empty_detail(ACTION_201604_FILE, 'Action 4')


# 上面简单的输出了下存在空值的文件中具体哪些列存在空值(True)，结果如下
# 
# * User
#     * age
#     * sex
#     * user_reg_tm
# * Action
#     * model_id
#     
# 接下来具体看看各文件中的空值情况:

# In[5]:


def empty_records(f_path, f_name, col_name):
    df_file = pd.read_csv(f_path,nrows=500000)
    missing = df_file[col_name].isnull().sum().sum()
    print('No. of missing {0} in {1} is {2}'.format(col_name, f_name, missing) )
    print ('percent: ', missing * 1.0 / df_file.shape[0])
empty_records(USER_FILE, 'User', 'age')
empty_records(USER_FILE, 'User', 'sex')
empty_records(USER_FILE, 'User', 'user_reg_tm')
empty_records(ACTION_201602_FILE, 'Action 2', 'model_id')
empty_records(ACTION_201603_FILE, 'Action 3', 'model_id')
empty_records(ACTION_201604_FILE, 'Action 4', 'model_id')


# 对比下数据集的记录数：
# 
# 文件|	文件说明 |	记录数
# --|:--:|--:
# 1. JData_User.csv       |用户数据集       |105,321个用户
# 2. JData_Comment.csv     |商品评论        |	558,552条记录
# 3. JData_Product.csv     |预测商品集合     |	24,187条记录
# 4. JData_Action_201602.csv |2月份行为交互记录  |	11,485,424条记录
# 5. JData_Action_201603.csv |3月份行为交互记录  |	25,916,378条记录
# 6. JData_Action_201604.csv |4月份行为交互记录  |	13,199,934条记录

# 两相对比结合前面输出的情况，针对不同数据进行不同处理
# 
# * 用户文件
#     * age,sex:先填充为对应的未知状态（-1|2），后续作为未知状态的值进一步分析和处理
#     * user_reg_tm:暂时不做处理
# * 行为文件
#     * model_id涉及数目接近一半，而且当前针对该特征没有很好的处理方法，待定
# 

# In[6]:


user = pd.read_csv(USER_FILE)
user['age'].fillna('-1', inplace=True)
user['sex'].fillna(2, inplace=True)


# In[7]:


pd.isnull(user).any()


# In[8]:


nan_reg_tm = user[user['user_reg_tm'].isnull()]


# In[9]:


print( len(user['sex'].unique()))
print (len(user['user_lv_cd'].unique()))


# In[10]:


prod = pd.read_csv(PRODUCT_FILE)
print (len(prod['a1'].unique()))
print (len(prod['a2'].unique()))
print (len(prod['a3'].unique()))
# print len(prod['a2'].unique())
print (len(prod['brand'].unique()))


# ## 3.7.未知记录

# 接下来看看各个文件中的未知记录占的比重

# In[11]:


print ('No. of unknown age user: {0} and the percent: {1} '.format(user[user['age']=='-1'].shape[0],
                                                                  user[user['age']=='-1'].shape[0]*1.0/user.shape[0]))
print ('No. of unknown sex user: {0} and the percent: {1} '.format(user[user['sex']==2].shape[0],
                                                                  user[user['sex']==2].shape[0]*1.0/user.shape[0]))


# In[12]:


def unknown_records(f_path, f_name, col_name):
    df_file = pd.read_csv(f_path)
    missing = df_file[df_file[col_name]==-1].shape[0]
    print ('No. of unknown {0} in {1} is {2}'.format(col_name, f_name, missing) )
    print ('percent: ', missing * 1.0 / df_file.shape[0])
    
unknown_records(PRODUCT_FILE, 'Product', 'a1')
unknown_records(PRODUCT_FILE, 'Product', 'a2')
unknown_records(PRODUCT_FILE, 'Product', 'a3')


# **小结一下:**
# 
# * 空值部分对3条用户的sex,age填充为未知值,注册时间不作处理，此外行为数据部分model_id待定: 43.2%,40.7%,39.0%
# * 未知值部分，用户age存在部分未知值: 13.7%，sex存在大量保密情况(超过一半) 52.0%
# * 商品中各个属性存在部分未知的情况，a1<a3<a2，分别为： 7.0%,16.7%,15.8%
# 

# ## 3.8 数据一致性验证

# 首先检查JData_User中的用户和JData_Action中的用户是否一致，保证行为数据中的所产生的行为均由用户数据中的用户产生（但是可能存在用户在行为数据中无行为）
# 
# 思路：利用pd.Merge连接sku 和 Action中的sku, 观察Action中的数据是否减少

# In[13]:


def user_action_check():
    df_user = pd.read_csv('data/JData_User.csv',nrows=500000)
    df_sku = df_user.ix[:,'user_id'].to_frame()
    df_month2 = pd.read_csv('data/JData_Action_201602.csv')
    print ('Is action of Feb. from User file? ', len(df_month2) == len(pd.merge(df_sku,df_month2)))
    df_month3 = pd.read_csv('data/JData_Action_201603.csv')
    print ('Is action of Mar. from User file? ', len(df_month3) == len(pd.merge(df_sku,df_month3)))
    df_month4 = pd.read_csv('data/JData_Action_201604.csv')
    print ('Is action of Apr. from User file? ', len(df_month4) == len(pd.merge(df_sku,df_month4)))
user_action_check()


# **结论：** User数据集中的用户和交互行为数据集中的用户完全一致
# 
# 根据merge前后的数据量比对，能保证Action中的用户ID是User中的ID的子集

# ## 3.9重复记录分析

# 除去各个数据文件中完全重复的记录,结果证明线上成绩反而大幅下降，可能解释是重复数据是有意义的，比如用户同时购买多件商品，同时添加多个数量的商品到购物车等…

# In[ ]:


def deduplicate(filepath, filename, newpath):
    df_file = pd.read_csv(filepath,nrows=500000)       
    before = df_file.shape[0]
    df_file.drop_duplicates(inplace=True)
    after = df_file.shape[0]
    n_dup = before-after
    print ('No. of duplicate records for ' + filename + ' is: ' + str(n_dup))
    if n_dup != 0:
        df_file.to_csv(newpath, index=None)
    else:
        print ('no duplicate records in ' + filename)


# In[ ]:


# deduplicate('data/JData_Action_201602.csv', 'Feb. action', 'data/JData_Action_201602_dedup.csv')
deduplicate('data/JData_Action_201603.csv', 'Mar. action', 'data/JData_Action_201603_dedup.csv')
deduplicate('data/JData_Action_201604.csv', 'Feb. action', 'data/JData_Action_201604_dedup.csv')
deduplicate('data/JData_Comment.csv', 'Comment', 'data/JData_Comment_dedup.csv')
deduplicate('data/JData_Product.csv', 'Product', 'data/JData_Product_dedup.csv')
deduplicate('data/JData_User.csv', 'User', 'data/JData_User_dedup.csv')


# In[28]:


df_month = pd.read_csv('data\JData_Action_201604.csv',nrows=500000)
df_month['time'] = pd.to_datetime(df_month['time'])
df_month.ix[df_month.time >= '2016-4-16']


# In[29]:


IsDuplicated = df_month.duplicated() 
df_d=df_month[IsDuplicated]
df_d.groupby('type').count()


# ## 3.10 检查是否存在注册时间在2016年-4月-15号之后的用户

# In[191]:


import pandas as pd
df_user = pd.read_csv('data\JData_User.csv',encoding='gbk')
df_user['user_reg_tm']=pd.to_datetime(df_user['user_reg_tm'])
df_user.ix[df_user.user_reg_tm  >= '2016-4-15'].head()


# 由于注册时间是京东系统错误造成，如果行为数据中没有在4月15号之后的数据的话，那么说明这些用户还是正常用户，并不需要删除。

# 结论：说明用户没有异常操作数据，所以这一批用户不删除

# ## 3.11 行为数据中的user_id为浮点型，进行INT类型转换

# In[47]:


# df_month = pd.read_csv('data\JData_Action_201602.csv',nrows=500000)
# print(df_month['user_id'].dtype)
# df_month['user_id'] = df_month['user_id'].apply(lambda x:int(x))
# print(df_month['user_id'].dtype)
# df_month.to_csv('data\JData_Action_201602.csv',index=None)
# df_month = pd.read_csv('data\JData_Action_201603.csv',nrows=500000)
# df_month['user_id'] = df_month['user_id'].apply(lambda x:int(x))
# print (df_month['user_id'].dtype)
# df_month.to_csv('data\JData_Action_201603.csv',index=None)
# df_month = pd.read_csv('data\JData_Action_201604.csv',nrows=500000)
# df_month['user_id'] = df_month['user_id'].apply(lambda x:int(x))
# df_month.to_csv('data\JData_Action_201604.csv',index=None)


# ## 3.12 按照星期对用户购买行为进行分析

# In[85]:


# 提取购买(type=4)的行为数据
def get_from_action_data(fname, chunk_size=1000000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)[
                ["user_id", "sku_id", "type", "time"]]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    df_ac = pd.concat(chunks, ignore_index=True)
    # type=4,为购买
    df_ac = df_ac[df_ac['type'] == 4]
    return df_ac[["user_id", "sku_id", "time"]]


# In[36]:


df_ac = []
df_ac.append(get_from_action_data(fname=ACTION_201602_FILE))
df_ac.append(get_from_action_data(fname=ACTION_201603_FILE))
df_ac.append(get_from_action_data(fname=ACTION_201604_FILE))
df_ac = pd.concat(df_ac, ignore_index=True)


# In[37]:


# 将time字段转换为datetime类型
df_ac['time'] = pd.to_datetime(df_ac['time'])
# 使用lambda匿名函数将时间time转换为星期(周一为1, 周日为７)
df_ac['time'] = df_ac['time'].apply(lambda x: x.weekday() + 1)


# In[66]:


# 周一到周日每天购买用户个数
df_user = df_ac.groupby('time')['user_id'].nunique()
df_user = df_user.to_frame().reset_index()
df_user.columns = ['weekday', 'user_num']


# In[39]:


# 周一到周日每天购买商品个数
df_item = df_ac.groupby('time')['sku_id'].nunique()
df_item = df_item.to_frame().reset_index()
df_item.columns = ['weekday', 'item_num']


# In[40]:


# 周一到周日每天购买记录个数
df_ui = df_ac.groupby('time', as_index=False).size()
df_ui = df_ui.to_frame().reset_index()
df_ui.columns = ['weekday', 'user_item_num']


# In[43]:


# 条形宽度
bar_width = 0.2
# 透明度
opacity = 0.4
plt.figure(figsize=(10,6))
plt.bar(df_user['weekday'], df_user['user_num'], bar_width, 
        alpha=opacity, color='c', label='user')
plt.bar(df_item['weekday']+bar_width, df_item['item_num'], 
        bar_width, alpha=opacity, color='g', label='item')
plt.bar(df_ui['weekday']+bar_width*2, df_ui['user_item_num'], 
        bar_width, alpha=opacity, color='m', label='user_item')
plt.xlabel('weekday')
plt.ylabel('number')
plt.title('A Week Purchase Table')
plt.xticks(df_user['weekday'] + bar_width * 3 / 2., (1,2,3,4,5,6,7))
plt.tight_layout() 
plt.legend(prop={'size':10})


# ## 3.13 一个月中各天购买量

# ### 3.13.1 2016年2月

# In[86]:


df_ac = get_from_action_data(fname=ACTION_201602_FILE)
# 将time字段转换为datetime类型并使用lambda匿名函数将时间time转换为天
df_ac['time'] = pd.to_datetime(df_ac['time']).apply(lambda x: x.day)


# In[87]:


df_user = df_ac.groupby('time')['user_id'].nunique()
df_user = df_user.to_frame().reset_index()
df_user.columns=['day','user_num']

df_item = df_ac.groupby('time')['sku_id'].nunique()
df_item = df_item.to_frame().reset_index()
df_item.columns = ['day', 'item_num']

df_ui = df_ac.groupby('time', as_index=False).size()
df_ui = df_ui.to_frame().reset_index()
df_ui.columns = ['day', 'user_item_num']


# In[89]:


# 条形宽度
bar_width = 0.2
# 透明度
opacity = 0.4
# 天数
day_range = range(1,len(df_user['day']) + 1, 1)
# 设置图片大小
plt.figure(figsize=(14,6))
plt.bar(df_user['day'], df_user['user_num'], bar_width, 
        alpha=opacity, color='c', label='user')
plt.bar(df_item['day']+bar_width, df_item['item_num'], 
        bar_width, alpha=opacity, color='g', label='item')
plt.bar(df_ui['day']+bar_width*2, df_ui['user_item_num'], 
        bar_width, alpha=opacity, color='m', label='user_item')
plt.xlabel('day')
plt.ylabel('number')
plt.title('February Purchase Table')
plt.xticks(df_user['day'] + bar_width * 3 / 2., day_range)
# plt.ylim(0, 80)
plt.tight_layout() 
plt.legend(prop={'size':9})


# ### 3.13.2 2016年3月

# In[90]:


df_ac = get_from_action_data(fname=ACTION_201603_FILE)
# 将time字段转换为datetime类型并使用lambda匿名函数将时间time转换为天
df_ac['time'] = pd.to_datetime(df_ac['time']).apply(lambda x: x.day)


# In[91]:


df_user = df_ac.groupby('time')['user_id'].nunique()
df_user = df_user.to_frame().reset_index()
df_user.columns = ['day', 'user_num']

df_item = df_ac.groupby('time')['sku_id'].nunique()
df_item = df_item.to_frame().reset_index()
df_item.columns = ['day', 'item_num']

df_ui = df_ac.groupby('time', as_index=False).size()
df_ui = df_ui.to_frame().reset_index()
df_ui.columns = ['day', 'user_item_num']


# In[92]:


# 条形宽度
bar_width = 0.2
# 透明度
opacity = 0.4
# 天数
day_range = range(1,len(df_user['day']) + 1, 1)
# 设置图片大小
plt.figure(figsize=(14,6))
plt.bar(df_user['day'], df_user['user_num'], bar_width, 
        alpha=opacity, color='c', label='user')
plt.bar(df_item['day']+bar_width, df_item['item_num'], 
        bar_width, alpha=opacity, color='g', label='item')
plt.bar(df_ui['day']+bar_width*2, df_ui['user_item_num'], 
        bar_width, alpha=opacity, color='m', label='user_item')
plt.xlabel('day')
plt.ylabel('number')
plt.title('March Purchase Table')
plt.xticks(df_user['day'] + bar_width * 3 / 2., day_range)
# plt.ylim(0, 80)
plt.tight_layout() 
plt.legend(prop={'size':9})


# **分析**：3月份14,15,16不知名节日，造成购物量剧增，总体来看，购物记录多于2月份

# ### 3.13.3 2016年4月

# In[93]:


df_ac = get_from_action_data(fname=ACTION_201604_FILE)
# 将time字段转换为datetime类型并使用lambda匿名函数将时间time转换为天
df_ac['time'] = pd.to_datetime(df_ac['time']).apply(lambda x: x.day)


# In[95]:


df_user = df_ac.groupby('time')['user_id'].nunique()
df_user = df_user.to_frame().reset_index()
df_user.columns = ['day', 'user_num']
df_item = df_ac.groupby('time')['sku_id'].nunique()
df_item = df_item.to_frame().reset_index()
df_item.columns = ['day', 'item_num']
df_ui = df_ac.groupby('time', as_index=False).size()
df_ui = df_ui.to_frame().reset_index()
df_ui.columns = ['day', 'user_item_num']


# In[96]:


# 条形宽度
bar_width = 0.2
# 透明度
opacity = 0.4
# 天数
day_range = range(1,len(df_user['day']) + 1, 1)
# 设置图片大小
plt.figure(figsize=(14,6))
plt.bar(df_user['day'], df_user['user_num'], bar_width, 
        alpha=opacity, color='c', label='user')
plt.bar(df_item['day']+bar_width, df_item['item_num'], 
        bar_width, alpha=opacity, color='g', label='item')
plt.bar(df_ui['day']+bar_width*2, df_ui['user_item_num'], 
        bar_width, alpha=opacity, color='m', label='user_item')
plt.xlabel('day')
plt.ylabel('number')
plt.title('April Purchase Table')
plt.xticks(df_user['day'] + bar_width * 3 / 2., day_range)
# plt.ylim(0, 80)
plt.tight_layout() 
plt.legend(prop={'size':9})


# ## 3.14 商品分类别销售统计

# ### 3.14.1 周一到周日各商品类别销售情况

# In[98]:


# 从行为记录中提取商品类别数据
def get_from_action_data(fname, chunk_size=1000000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)[
                ["cate", "brand", "type", "time"]]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    df_ac = pd.concat(chunks, ignore_index=True)
    # type=4,为购买
    df_ac = df_ac[df_ac['type'] == 4]
    return df_ac[["cate", "brand", "type", "time"]]


# In[99]:


df_ac = []
df_ac.append(get_from_action_data(fname=ACTION_201602_FILE))
df_ac.append(get_from_action_data(fname=ACTION_201603_FILE))
df_ac.append(get_from_action_data(fname=ACTION_201604_FILE))
df_ac = pd.concat(df_ac, ignore_index=True)


# In[100]:


# 将time字段转换为datetime类型
df_ac['time'] = pd.to_datetime(df_ac['time'])
# 使用lambda匿名函数将时间time转换为星期(周一为1, 周日为７)
df_ac['time'] = df_ac['time'].apply(lambda x: x.weekday() + 1)


# In[101]:


# 观察有几个类别商品
df_ac.groupby(df_ac['cate']).count()


# In[104]:


# 周一到周日每天购买商品类别4的数量统计
df_product = df_ac['brand'].groupby([df_ac['time'],df_ac['cate']]).count()
df_product=df_product.unstack()
df_product.plot(kind='bar',title='Cate Purchase Table in a Week',figsize=(16,6))


# **分析**：星期二买类别8的最多，星期天最少。

# ### 3.14.2 每月各类商品销售情况（只关注商品8）

# #### 3.14.2.1 2016年2，3，4月

# In[105]:


df_ac2 = get_from_action_data(fname=ACTION_201602_FILE)
# 将time字段转换为datetime类型并使用lambda匿名函数将时间time转换为天
df_ac2['time'] = pd.to_datetime(df_ac2['time']).apply(lambda x: x.day)

df_ac3 = get_from_action_data(fname=ACTION_201603_FILE)
# 将time字段转换为datetime类型并使用lambda匿名函数将时间time转换为天
df_ac3['time'] = pd.to_datetime(df_ac3['time']).apply(lambda x: x.day)

df_ac4 = get_from_action_data(fname=ACTION_201604_FILE)
# 将time字段转换为datetime类型并使用lambda匿名函数将时间time转换为天
df_ac4['time'] = pd.to_datetime(df_ac4['time']).apply(lambda x: x.day)


# In[107]:


dc_cate2 = df_ac2[df_ac2['cate']==8]
dc_cate2 = dc_cate2['brand'].groupby(dc_cate2['time']).count()
dc_cate2 = dc_cate2.to_frame().reset_index()
dc_cate2.columns = ['day', 'product_num']

dc_cate3 = df_ac3[df_ac3['cate']==8]
dc_cate3 = dc_cate3['brand'].groupby(dc_cate3['time']).count()
dc_cate3 = dc_cate3.to_frame().reset_index()
dc_cate3.columns = ['day', 'product_num']

dc_cate4 = df_ac4[df_ac4['cate']==8]
dc_cate4 = dc_cate4['brand'].groupby(dc_cate4['time']).count()
dc_cate4 = dc_cate4.to_frame().reset_index()
dc_cate4.columns = ['day', 'product_num']


# In[108]:


# 条形宽度
bar_width = 0.2
# 透明度
opacity = 0.4
# 天数
day_range = range(1,len(dc_cate3['day']) + 1, 1)
# 设置图片大小
plt.figure(figsize=(14,6))
plt.bar(dc_cate2['day'], dc_cate2['product_num'], bar_width, 
        alpha=opacity, color='c', label='February')
plt.bar(dc_cate3['day']+bar_width, dc_cate3['product_num'], 
        bar_width, alpha=opacity, color='g', label='March')
plt.bar(dc_cate4['day']+bar_width*2, dc_cate4['product_num'], 
        bar_width, alpha=opacity, color='m', label='April')
plt.xlabel('day')
plt.ylabel('number')
plt.title('Cate-8 Purchase Table')
plt.xticks(dc_cate3['day'] + bar_width * 3 / 2., day_range)
# plt.ylim(0, 80)
plt.tight_layout() 
plt.legend(prop={'size':9})


# **分析**：2月份对类别8商品的购买普遍偏低，3，4月份普遍偏高，3月15日购买极其多！可以对比3月份的销售记录，发现类别8将近占了3月15日总销售的一半！同时发现，3,4月份类别8销售记录在前半个月特别相似，除了4月8号，9号和3月15号。

# # 4. 特征工程

# In[19]:


df = pd.read_csv(USER_FILE,header=0,encoding='gbk',nrows=500000)
df.isnull().sum()


# In[41]:


comment_date = [
    "2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29",
    "2016-03-07", "2016-03-14", "2016-03-21", "2016-03-28", "2016-04-04",
    "2016-04-11", "2016-04-15"]

def get_action_data(FILE):
    reader = pd.read_csv(FILE,encoding='gbk',header=0,iterator=True)
    chunks=[]
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(1000000)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print('Iteration Stopped!!')
    df_ac = pd.concat(chunks, ignore_index=True)
    return df_ac
action_1_path =    ACTION_201602_FILE
action_2_path =    ACTION_201603_FILE
action_3_path =    ACTION_201604_FILE

def get_actions_1():
    action = pd.read_csv(action_1_path,header=0,encoding='gbk',nrows=1000000)
    action = action.sample(n = 800000)
#     action = get_action_data(action_1_path)
    return action
def get_actions_2():
    action2 = pd.read_csv(action_2_path,header=0,encoding='gbk',nrows=1000000)
#     action2 = get_action_data(action_1_path)
    action2 = action2.sample(n = 800000)
    return action2
def get_actions_3():
    action3 = pd.read_csv(action_3_path,header=0,encoding='gbk',nrows=1000000)
#     action3 = get_action_data(action_1_path)
    action3 = action3.sample(n = 800000)
    return action3
# 读取并拼接所有行为记录文件
def get_all_action():
    action_1 = get_actions_1()
    action_2 = get_actions_2()
    action_3 = get_actions_3()
    actions = pd.concat([action_1, action_2, action_3]) # type: pd.DataFrame
#     actions = pd.read_csv(action_path)
    return actions
# 获取某个时间段的行为记录
def get_actions(start_date, end_date, all_actions):
    """
    :param start_date:
    :param end_date:
    :return: actions: pd.Dataframe
    """
    actions = all_actions[(all_actions.time >= start_date) & (all_actions.time < end_date)].copy()
    return actions


# ## 4.1. 用户特征

# ### 4.1.1 用户基本特征

# 获取基本的用户特征，基于用户本身属性多为类别特征的特点，对age,sex,usr_lv_cd进行独热编码操作，对于用户注册时间暂时不处理

# In[9]:


user_path = USER_FILE

from sklearn import preprocessing
def get_basic_user_feat():
    # 针对年龄的中文字符问题处理，首先是读入的时候编码，填充空值，然后将其数值化，最后独热编码，此外对于sex也进行了数值类型转换
    user = pd.read_csv(user_path, encoding='gbk',header=0)
    user['age'].fillna('-1', inplace=True)
    user['sex'].fillna(2, inplace=True)
    user['sex'] = user['sex'].astype(int)    
    user['age'] = user['age'].astype(int)
    le = preprocessing.LabelEncoder()    
    age_df = le.fit_transform(user['age'])
#     print list(le.classes_)
    age_df = pd.get_dummies(age_df, prefix='age')
    sex_df = pd.get_dummies(user['sex'], prefix='sex')
    user_lv_df = pd.get_dummies(user['user_lv_cd'], prefix='user_lv_cd')
    user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df], axis=1)
    return user


# In[7]:


get_basic_user_feat().info()


# ## 4.2. 商品特征

# ### 4.2.1.  商品基本特征

# 根据商品文件获取基本的特征，针对属性a1,a2,a3进行独热编码，商品类别和品牌直接作为特征

# In[10]:


product_path = PRODUCT_FILE
def get_basic_product_feat():
    product = pd.read_csv(product_path,encoding='gbk',header=0)
    attr1_df = pd.get_dummies(product["a1"], prefix="a1")
    attr2_df = pd.get_dummies(product["a2"], prefix="a2")
    attr3_df = pd.get_dummies(product["a3"], prefix="a3")
    product = pd.concat([product[['sku_id', 'cate', 'brand']], attr1_df, attr2_df, attr3_df], axis=1)
    return product


# In[9]:


get_basic_product_feat().info()


# ## 4.3. 评论特征
# 
# * 分时间段
# * 对评论数进行独热编码
# 

# In[11]:


comment_path = COMMENT_FILE
def get_comments_product_feat(end_date):
    print('商品评论特征(get_comments_product_feat):end_date:'+end_date)
    comments = pd.read_csv(comment_path,header=0,encoding='gbk')
    comment_date_end = end_date
    comment_date_begin = comment_date[0]
    for date in reversed(comment_date):
        if date < comment_date_end:
            comment_date_begin = date
            break
    comments = comments[comments.dt==comment_date_begin]
    df = pd.get_dummies(comments['comment_num'], prefix='comment_num')
    # 为了防止某个时间段不具备评论数为0的情况（测试集出现过这种情况）
    for i in range(0, 5):
        if 'comment_num_' + str(i) not in df.columns:
            df['comment_num_' + str(i)] = 0
    df = df[['comment_num_0', 'comment_num_1', 'comment_num_2', 'comment_num_3', 'comment_num_4']]
    
    comments = pd.concat([comments, df], axis=1) # type: pd.DataFrame
        #del comments['dt']
        #del comments['comment_num']
    comments = comments[['sku_id', 'has_bad_comment', 'bad_comment_rate','comment_num_0', 'comment_num_1', 
                         'comment_num_2', 'comment_num_3', 'comment_num_4']]
    return comments


# In[11]:


get_comments_product_feat('2016-03-01').info()


# ## 4.4.  行为特征
# 
# * 分时间段
# * 对行为类别进行独热编码
# * 分别按照用户-类别行为分组和用户-类别-商品行为分组统计，然后计算
#     * 用户对同类别下其他商品的行为计数
#     * 针对用户对同类别下目标商品的行为计数与该时间段的行为均值作差
# 
# 
# 

# In[42]:


all_actions = get_all_action()


# In[22]:




# In[12]:


def get_action_feat(start_date, end_date, all_actions, i):
    print('行为特征(get_action_feat):{}=>{}:{}'.format(start_date,end_date,i))
    actions = get_actions(start_date, end_date, all_actions)
    actions = actions[['user_id', 'sku_id', 'cate','type']]
    # 不同时间累积的行为计数（3,5,7,10,15,21,30）
    df = pd.get_dummies(actions['type'], prefix='action_before_%s' %i)
    before_date = 'action_before_%s' %i
    actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
    # 分组统计，用户-类别-商品,不同用户对不同类别下商品的行为计数
    actions = actions.groupby(['user_id', 'sku_id','cate'], as_index=False).sum()
    # 分组统计，用户-类别，不同用户对不同商品类别的行为计数
    user_cate = actions.groupby(['user_id','cate'], as_index=False).sum()
    del user_cate['sku_id']
    del user_cate['type']
    actions = pd.merge(actions, user_cate, how='left', on=['user_id','cate'])
    #本类别下其他商品点击量
    # 前述两种分组含有相同名称的不同行为的计数，系统会自动针对名称调整添加后缀,x,y，所以这里作差统计的是同一类别下其他商品的行为计数
    actions[before_date+'_1_y'] = actions[before_date+'_1_y'] - actions[before_date+'_1_x']
    actions[before_date+'_2_y'] = actions[before_date+'_2_y'] - actions[before_date+'_2_x']
    actions[before_date+'_3_y'] = actions[before_date+'_3_y'] - actions[before_date+'_3_x']
    actions[before_date+'_4_y'] = actions[before_date+'_4_y'] - actions[before_date+'_4_x']
    actions[before_date+'_5_y'] = actions[before_date+'_5_y'] - actions[before_date+'_5_x']
    actions[before_date+'_6_y'] = actions[before_date+'_6_y'] - actions[before_date+'_6_x']
    # 统计用户对不同类别下商品计数与该类别下商品行为计数均值（对时间）的差值
    actions[before_date+'minus_mean_1'] = actions[before_date+'_1_x'] - (actions[before_date+'_1_x']/i)
    actions[before_date+'minus_mean_2'] = actions[before_date+'_2_x'] - (actions[before_date+'_2_x']/i)
    actions[before_date+'minus_mean_3'] = actions[before_date+'_3_x'] - (actions[before_date+'_3_x']/i)
    actions[before_date+'minus_mean_4'] = actions[before_date+'_4_x'] - (actions[before_date+'_4_x']/i)
    actions[before_date+'minus_mean_5'] = actions[before_date+'_5_x'] - (actions[before_date+'_5_x']/i)
    actions[before_date+'minus_mean_6'] = actions[before_date+'_6_x'] - (actions[before_date+'_6_x']/i)
    del actions['type']
    # 保留cate特征
#     del actions['cate']
    return actions


# In[14]:


get_action_feat('2016-02-01','2016-03-01',all_actions,3).info()


# In[6]:


df1 = get_action_feat('2016-02-01','2016-03-01',all_actions,3)
df2 = get_action_feat('2016-02-01','2016-03-01',all_actions,5)
df3 = get_action_feat('2016-02-01','2016-03-01',all_actions,7)
print(df1.shape,df2.shape,df3.shape)


# In[43]:


df1 = get_action_feat('2016-04-01','2016-04-04',all_actions,3)
df2 = get_action_feat('2016-03-30','2016-04-04',all_actions,5)
df3 = get_action_feat('2016-03-28','2016-04-04',all_actions,75)
df4 = get_action_feat('2016-03-25','2016-04-04',all_actions,10)
df5 = get_action_feat('2016-03-20','2016-04-04',all_actions,15)
df6 = get_action_feat('2016-03-14','2016-04-04',all_actions,21)
df7 = get_action_feat('2016-03-05','2016-04-04',all_actions,30)
print(df1.shape,df2.shape,df3.shape,df4.shape,df5.shape,df6.shape,df7.shape)


# In[ ]:


df1 = get_action_feat('2016-04-02','2016-04-05',all_actions,3)
df2 = get_action_feat('2016-03-31','2016-04-05',all_actions,5)
df3 = get_action_feat('2016-03-29','2016-04-05',all_actions,75)
df4 = get_action_feat('2016-03-26','2016-04-05',all_actions,10)
df5 = get_action_feat('2016-03-21','2016-04-05',all_actions,15)
df6 = get_action_feat('2016-03-15','2016-04-05',all_actions,21)
df7 = get_action_feat('2016-03-06','2016-04-05',all_actions,30)
print(df1.shape,df2.shape,df3.shape,df4.shape,df5.shape,df6.shape,df7.shape)


# ## 4.5.用户-行为

# ### 4.5.1.  累积用户特征
# 
# * 分时间段
# * 用户不同行为的
#     * 购买转化率
#     * 均值
#     * 标准差
# 

# In[13]:


def get_accumulate_user_feat(end_date, all_actions, day):
    start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=day)
    start_date = start_date.strftime('%Y-%m-%d')
    print('累积用户特征(get_accumulate_user_feat):{}=>{}:{}'.format(start_date,end_date,day))
    before_date = 'user_action_%s' % day
    feature = [
        'user_id', before_date + '_1', before_date + '_2', before_date + '_3',
        before_date + '_4', before_date + '_5', before_date + '_6',
        before_date + '_1_ratio', before_date + '_2_ratio',
        before_date + '_3_ratio', before_date + '_5_ratio',
        before_date + '_6_ratio', before_date + '_1_mean',
        before_date + '_2_mean', before_date + '_3_mean',
        before_date + '_4_mean', before_date + '_5_mean',
        before_date + '_6_mean', before_date + '_1_std',
        before_date + '_2_std', before_date + '_3_std', before_date + '_4_std',
        before_date + '_5_std', before_date + '_6_std'
    ]
    actions = get_actions(start_date, end_date, all_actions) #根据时间获取行为数据
    if len(actions['type'].unique()) != 6:
        actions['type'] = np.random.randint(1,7,size=(len(actions),))
    df = pd.get_dummies(actions['type'], prefix=before_date)
    actions['date'] = pd.to_datetime(actions['time']).apply(lambda x: x.date())
    actions = pd.concat([actions[['user_id', 'date']], df], axis=1)
    # 分组统计，用户不同日期的行为计算标准差
    actions_date = actions.groupby(['user_id', 'date']).sum()
    actions_date = actions_date.unstack()
    actions_date.fillna(0, inplace=True)
    action_1 = np.std(actions_date[before_date + '_1'], axis=1)# 这段时间内type=1的std
    action_1 = action_1.to_frame()
    action_1.columns = [before_date + '_1_std']
    action_2 = np.std(actions_date[before_date + '_2'], axis=1)
    action_2 = action_2.to_frame()
    action_2.columns = [before_date + '_2_std']
    action_3 = np.std(actions_date[before_date + '_3'], axis=1)
    action_3 = action_3.to_frame()
    action_3.columns = [before_date + '_3_std']
    action_4 = np.std(actions_date[before_date + '_4'], axis=1)
    action_4 = action_4.to_frame()
    action_4.columns = [before_date + '_4_std']
    action_5 = np.std(actions_date[before_date + '_5'], axis=1)
    action_5 = action_5.to_frame()
    action_5.columns = [before_date + '_5_std']
    action_6 = np.std(actions_date[before_date + '_6'], axis=1)
    action_6 = action_6.to_frame()
    action_6.columns = [before_date + '_6_std']
    actions_date = pd.concat(
        [action_1, action_2, action_3, action_4, action_5, action_6], axis=1)
    actions_date['user_id'] = actions_date.index
    # 分组统计，按用户分组，统计用户各项行为的转化率、均值
    actions = actions.groupby(['user_id'], as_index=False).sum()
#     days_interal = (datetime.strptime(end_date, '%Y-%m-%d') -
#                     datetime.strptime(start_date, '%Y-%m-%d')).days
    # 转化率
#     actions[before_date + '_1_ratio'] = actions[before_date +
#                                                 '_4'] / actions[before_date +
#                                                                 '_1']
#     actions[before_date + '_2_ratio'] = actions[before_date +
#                                                 '_4'] / actions[before_date +
#                                                                 '_2']
#     actions[before_date + '_3_ratio'] = actions[before_date +
#                                                 '_4'] / actions[before_date +
#                                                                 '_3']
#     actions[before_date + '_5_ratio'] = actions[before_date +
#                                                 '_4'] / actions[before_date +
#                                                                 '_5']
#     actions[before_date + '_6_ratio'] = actions[before_date +
#                                                 '_4'] / actions[before_date +
#                                                                 '_6']
    # 类型为4的总次数/各自总次数，即各种转化率
    actions[before_date + '_1_ratio'] =  np.log(1 + actions[before_date + '_4']) - np.log(1 + actions[before_date +'_1'])
    actions[before_date + '_2_ratio'] =  np.log(1 + actions[before_date + '_4']) - np.log(1 + actions[before_date +'_2'])
    actions[before_date + '_3_ratio'] =  np.log(1 + actions[before_date + '_4']) - np.log(1 + actions[before_date +'_3'])
    actions[before_date + '_5_ratio'] =  np.log(1 + actions[before_date + '_4']) - np.log(1 + actions[before_date +'_5'])
    actions[before_date + '_6_ratio'] =  np.log(1 + actions[before_date + '_4']) - np.log(1 + actions[before_date +'_6'])
    # 均值
    actions[before_date + '_1_mean'] = actions[before_date + '_1'] / day
    actions[before_date + '_2_mean'] = actions[before_date + '_2'] / day
    actions[before_date + '_3_mean'] = actions[before_date + '_3'] / day
    actions[before_date + '_4_mean'] = actions[before_date + '_4'] / day
    actions[before_date + '_5_mean'] = actions[before_date + '_5'] / day
    actions[before_date + '_6_mean'] = actions[before_date + '_6'] / day
    actions = pd.merge(actions, actions_date, how='left', on='user_id')
    actions = actions[feature]
    return actions


# In[32]:


actions = get_actions('2016-02-01', '2016-03-01', all_actions)
df = pd.get_dummies(actions['type'], prefix='before_date')
actions['date'] = pd.to_datetime(actions['time']).apply(lambda x: x.date())
actions = pd.concat([actions[['user_id', 'date']], df], axis=1)


# In[36]:


df2 = actions.groupby('user_id').sum()
df2.head()


# In[18]:


print(actions.info(),actions.head())


# In[18]:


df = pd.get_dummies(actions['type'], prefix='user_action_3')
actions['date'] = pd.to_datetime(actions['time']).apply(lambda x: x.date())
actions = pd.concat([actions[['user_id', 'date']], df], axis=1)
# 分组统计，用户不同日期的行为计算标准差
actions_date = actions.groupby(['user_id', 'date']).sum()
actions_date = actions_date.unstack()


# In[20]:


user_acc = get_accumulate_user_feat('2016-03-01',all_actions,30)
user_acc.info()


# ### 4.5.2. 用户近期行为特征

# 在上面针对用户进行累积特征提取的基础上，分别提取用户近一个月、近三天的特征，然后提取一个月内用户除去最近三天的行为占据一个月的行为的比重

# In[14]:


def get_recent_user_feat(end_date, all_actions):
    print('用户近期行为特征(get_recent_user_feat,end_date){}'.format(end_date))
    actions_3 = get_accumulate_user_feat(end_date, all_actions, 3) #由于取的是前多少行数据，故没有取到所有的type数据
    actions_30 = get_accumulate_user_feat(end_date, all_actions, 30)
    actions = pd.merge(actions_3, actions_30, how ='left', on='user_id')
    del actions_3
    del actions_30
    
    actions['recent_action1'] =  np.log(1 + actions['user_action_30_1']-actions['user_action_3_1']) - np.log(1 + actions['user_action_30_1'])
    actions['recent_action2'] =  np.log(1 + actions['user_action_30_2']-actions['user_action_3_2']) - np.log(1 + actions['user_action_30_2'])
    actions['recent_action3'] =  np.log(1 + actions['user_action_30_3']-actions['user_action_3_3']) - np.log(1 + actions['user_action_30_3'])
    actions['recent_action4'] =  np.log(1 + actions['user_action_30_4']-actions['user_action_3_4']) - np.log(1 + actions['user_action_30_4'])
    actions['recent_action5'] =  np.log(1 + actions['user_action_30_5']-actions['user_action_3_5']) - np.log(1 + actions['user_action_30_5'])
    actions['recent_action6'] =  np.log(1 + actions['user_action_30_6']-actions['user_action_3_6']) - np.log(1 + actions['user_action_30_6'])
    
#     actions['recent_action1'] = (actions['user_action_30_1']-actions['user_action_3_1'])/actions['user_action_30_1']
#     actions['recent_action2'] = (actions['user_action_30_2']-actions['user_action_3_2'])/actions['user_action_30_2']
#     actions['recent_action3'] = (actions['user_action_30_3']-actions['user_action_3_3'])/actions['user_action_30_3']
#     actions['recent_action4'] = (actions['user_action_30_4']-actions['user_action_3_4'])/actions['user_action_30_4']
#     actions['recent_action5'] = (actions['user_action_30_5']-actions['user_action_3_5'])/actions['user_action_30_5']
#     actions['recent_action6'] = (actions['user_action_30_6']-actions['user_action_3_6'])/actions['user_action_30_6']
    
    return actions


# In[22]:


get_recent_user_feat('2016-03-01',all_actions).info()


# ### 4.5.3. 用户对同类别下各种商品的行为
# 
# * 用户对各个类别的各项行为操作统计
# * 用户对各个类别操作行为统计占对所有类别操作行为统计的比重
# 

# In[15]:


#增加了用户对不同类别的交互特征
def get_user_cate_feature(start_date, end_date, all_actions):
    print('用户对同类别下各种商品的行为(get_user_cate_feature):{}=>{}'.format(start_date,end_date))
    actions = get_actions(start_date, end_date, all_actions)
    actions = actions[['user_id', 'cate', 'type']]
    df = pd.get_dummies(actions['type'], prefix='type')
    actions = pd.concat([actions[['user_id', 'cate']], df], axis=1)
    actions = actions.groupby(['user_id', 'cate']).sum()
    actions = actions.unstack()
    actions.columns = actions.columns.swaplevel(0, 1)
    actions.columns = actions.columns.droplevel()
    actions.columns = [
        'cate_4_type1', 'cate_5_type1', 'cate_6_type1', 'cate_7_type1',
        'cate_8_type1', 'cate_9_type1', 'cate_10_type1', 'cate_11_type1',
        'cate_4_type2', 'cate_5_type2', 'cate_6_type2', 'cate_7_type2',
        'cate_8_type2', 'cate_9_type2', 'cate_10_type2', 'cate_11_type2',
        'cate_4_type3', 'cate_5_type3', 'cate_6_type3', 'cate_7_type3',
        'cate_8_type3', 'cate_9_type3', 'cate_10_type3', 'cate_11_type3',
        'cate_4_type4', 'cate_5_type4', 'cate_6_type4', 'cate_7_type4',
        'cate_8_type4', 'cate_9_type4', 'cate_10_type4', 'cate_11_type4',
        'cate_4_type5', 'cate_5_type5', 'cate_6_type5', 'cate_7_type5',
        'cate_8_type5', 'cate_9_type5', 'cate_10_type5', 'cate_11_type5',
        'cate_4_type6', 'cate_5_type6', 'cate_6_type6', 'cate_7_type6',
        'cate_8_type6', 'cate_9_type6', 'cate_10_type6', 'cate_11_type6'
    ]
    actions = actions.fillna(0)
    actions['cate_action_sum'] = actions.sum(axis=1)
    actions['cate8_percentage'] = (
        actions['cate_8_type1'] + actions['cate_8_type2'] +
        actions['cate_8_type3'] + actions['cate_8_type4'] +
        actions['cate_8_type5'] + actions['cate_8_type6']
    ) / actions['cate_action_sum']
    actions['cate4_percentage'] = (
        actions['cate_4_type1'] + actions['cate_4_type2'] +
        actions['cate_4_type3'] + actions['cate_4_type4'] +
        actions['cate_4_type5'] + actions['cate_4_type6']
    ) / actions['cate_action_sum']
    actions['cate5_percentage'] = (
        actions['cate_5_type1'] + actions['cate_5_type2'] +
        actions['cate_5_type3'] + actions['cate_5_type4'] +
        actions['cate_5_type5'] + actions['cate_5_type6']
    ) / actions['cate_action_sum']
    actions['cate6_percentage'] = (
        actions['cate_6_type1'] + actions['cate_6_type2'] +
        actions['cate_6_type3'] + actions['cate_6_type4'] +
        actions['cate_6_type5'] + actions['cate_6_type6']
    ) / actions['cate_action_sum']
    actions['cate7_percentage'] = (
        actions['cate_7_type1'] + actions['cate_7_type2'] +
        actions['cate_7_type3'] + actions['cate_7_type4'] +
        actions['cate_7_type5'] + actions['cate_7_type6']
    ) / actions['cate_action_sum']
    actions['cate9_percentage'] = (
        actions['cate_9_type1'] + actions['cate_9_type2'] +
        actions['cate_9_type3'] + actions['cate_9_type4'] +
        actions['cate_9_type5'] + actions['cate_9_type6']
    ) / actions['cate_action_sum']
    actions['cate10_percentage'] = (
        actions['cate_10_type1'] + actions['cate_10_type2'] +
        actions['cate_10_type3'] + actions['cate_10_type4'] +
        actions['cate_10_type5'] + actions['cate_10_type6']
    ) / actions['cate_action_sum']
    actions['cate11_percentage'] = (
        actions['cate_11_type1'] + actions['cate_11_type2'] +
        actions['cate_11_type3'] + actions['cate_11_type4'] +
        actions['cate_11_type5'] + actions['cate_11_type6']
    ) / actions['cate_action_sum']
    actions['cate8_type1_percentage'] = np.log(
        1 + actions['cate_8_type1']) - np.log(
            1 + actions['cate_8_type1'] + actions['cate_4_type1'] +
            actions['cate_5_type1'] + actions['cate_6_type1'] +
            actions['cate_7_type1'] + actions['cate_9_type1'] +
            actions['cate_10_type1'] + actions['cate_11_type1'])
    actions['cate8_type2_percentage'] = np.log(
        1 + actions['cate_8_type2']) - np.log(
            1 + actions['cate_8_type2'] + actions['cate_4_type2'] +
            actions['cate_5_type2'] + actions['cate_6_type2'] +
            actions['cate_7_type2'] + actions['cate_9_type2'] +
            actions['cate_10_type2'] + actions['cate_11_type2'])
    actions['cate8_type3_percentage'] = np.log(
        1 + actions['cate_8_type3']) - np.log(
            1 + actions['cate_8_type3'] + actions['cate_4_type3'] +
            actions['cate_5_type3'] + actions['cate_6_type3'] +
            actions['cate_7_type3'] + actions['cate_9_type3'] +
            actions['cate_10_type3'] + actions['cate_11_type3'])
    actions['cate8_type4_percentage'] = np.log(
        1 + actions['cate_8_type4']) - np.log(
            1 + actions['cate_8_type4'] + actions['cate_4_type4'] +
            actions['cate_5_type4'] + actions['cate_6_type4'] +
            actions['cate_7_type4'] + actions['cate_9_type4'] +
            actions['cate_10_type4'] + actions['cate_11_type4'])
    actions['cate8_type5_percentage'] = np.log(
        1 + actions['cate_8_type5']) - np.log(
            1 + actions['cate_8_type5'] + actions['cate_4_type5'] +
            actions['cate_5_type5'] + actions['cate_6_type5'] +
            actions['cate_7_type5'] + actions['cate_9_type5'] +
            actions['cate_10_type5'] + actions['cate_11_type5'])
    actions['cate8_type6_percentage'] = np.log(
        1 + actions['cate_8_type6']) - np.log(
            1 + actions['cate_8_type6'] + actions['cate_4_type6'] +
            actions['cate_5_type6'] + actions['cate_6_type6'] +
            actions['cate_7_type6'] + actions['cate_9_type6'] +
            actions['cate_10_type6'] + actions['cate_11_type6'])
    actions['user_id'] = actions.index
    actions = actions[[
        'user_id', 'cate8_percentage', 'cate4_percentage', 'cate5_percentage',
        'cate6_percentage', 'cate7_percentage', 'cate9_percentage',
        'cate10_percentage', 'cate11_percentage', 'cate8_type1_percentage',
        'cate8_type2_percentage', 'cate8_type3_percentage',
        'cate8_type4_percentage', 'cate8_type5_percentage',
        'cate8_type6_percentage'
    ]]
    return actions


# In[24]:


get_user_cate_feature('2016-02-01','2016-03-01',all_actions).info()


# ## 4.6. 商品-行为

# ### 4.6.1.  累积商品特征
# 
# * 分时间段
# * 针对商品的不同行为的
#     * 购买转化率
#     * 均值
#     * 标准差
# 

# In[16]:


def get_accumulate_product_feat(start_date, end_date, all_actions):
    print('累积商品特征(get_accumulate_product_feat):{}=>{}'.format(start_date,end_date))
    feature = [
        'sku_id', 'product_action_1', 'product_action_2',
        'product_action_3', 'product_action_4',
        'product_action_5', 'product_action_6',
        'product_action_1_ratio', 'product_action_2_ratio',
        'product_action_3_ratio', 'product_action_5_ratio',
        'product_action_6_ratio', 'product_action_1_mean',
        'product_action_2_mean', 'product_action_3_mean',
        'product_action_4_mean', 'product_action_5_mean',
        'product_action_6_mean', 'product_action_1_std',
        'product_action_2_std', 'product_action_3_std', 'product_action_4_std',
        'product_action_5_std', 'product_action_6_std'
    ]
    actions = get_actions(start_date, end_date, all_actions)
    df = pd.get_dummies(actions['type'], prefix='product_action')
    # 按照商品-日期分组，计算某个时间段该商品的各项行为的标准差
    actions['date'] = pd.to_datetime(actions['time']).apply(lambda x: x.date())
    actions = pd.concat([actions[['sku_id', 'date']], df], axis=1)
    actions_date = actions.groupby(['sku_id', 'date']).sum()
    actions_date = actions_date.unstack()
    actions_date.fillna(0, inplace=True)
    action_1 = np.std(actions_date['product_action_1'], axis=1)
    action_1 = action_1.to_frame()
    action_1.columns = ['product_action_1_std']
    action_2 = np.std(actions_date['product_action_2'], axis=1)
    action_2 = action_2.to_frame()
    action_2.columns = ['product_action_2_std']
    action_3 = np.std(actions_date['product_action_3'], axis=1)
    action_3 = action_3.to_frame()
    action_3.columns = ['product_action_3_std']
    action_4 = np.std(actions_date['product_action_4'], axis=1)
    action_4 = action_4.to_frame()
    action_4.columns = ['product_action_4_std']
    action_5 = np.std(actions_date['product_action_5'], axis=1)
    action_5 = action_5.to_frame()
    action_5.columns = ['product_action_5_std']
    action_6 = np.std(actions_date['product_action_6'], axis=1)
    action_6 = action_6.to_frame()
    action_6.columns = ['product_action_6_std']
    actions_date = pd.concat(
        [action_1, action_2, action_3, action_4, action_5, action_6], axis=1)
    actions_date['sku_id'] = actions_date.index
    actions = actions.groupby(['sku_id'], as_index=False).sum()
    days_interal = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
    # 针对商品分组，计算购买转化率
#     actions['product_action_1_ratio'] = actions['product_action_4'] / actions[
#         'product_action_1']
#     actions['product_action_2_ratio'] = actions['product_action_4'] / actions[
#         'product_action_2']
#     actions['product_action_3_ratio'] = actions['product_action_4'] / actions[
#         'product_action_3']
#     actions['product_action_5_ratio'] = actions['product_action_4'] / actions[
#         'product_action_5']
#     actions['product_action_6_ratio'] = actions['product_action_4'] / actions[
#         'product_action_6']
    actions['product_action_1_ratio'] =  np.log(1 + actions['product_action_4']) - np.log(1 + actions['product_action_1'])
    actions['product_action_2_ratio'] =  np.log(1 + actions['product_action_4']) - np.log(1 + actions['product_action_2'])
    actions['product_action_3_ratio'] =  np.log(1 + actions['product_action_4']) - np.log(1 + actions['product_action_3'])
    actions['product_action_5_ratio'] =  np.log(1 + actions['product_action_4']) - np.log(1 + actions['product_action_5'])
    actions['product_action_6_ratio'] =  np.log(1 + actions['product_action_4']) - np.log(1 + actions['product_action_6'])
    # 计算各种行为的均值
    actions['product_action_1_mean'] = actions[
        'product_action_1'] / days_interal
    actions['product_action_2_mean'] = actions[
        'product_action_2'] / days_interal
    actions['product_action_3_mean'] = actions[
        'product_action_3'] / days_interal
    actions['product_action_4_mean'] = actions[
        'product_action_4'] / days_interal
    actions['product_action_5_mean'] = actions[
        'product_action_5'] / days_interal
    actions['product_action_6_mean'] = actions[
        'product_action_6'] / days_interal
    actions = pd.merge(actions, actions_date, how='left', on='sku_id')
    actions = actions[feature]
    return actions


# In[26]:


get_accumulate_product_feat('2016-02-01','2016-03-01',all_actions).info()


# ## 4.7. 类别特征

# 分时间段下各个商品类别的
# 
# * 购买转化率
# * 标准差
# * 均值
# 

# In[17]:


def get_accumulate_cate_feat(start_date, end_date, all_actions):
    feature = ['cate','cate_action_1', 'cate_action_2', 'cate_action_3', 'cate_action_4', 'cate_action_5', 
               'cate_action_6', 'cate_action_1_ratio', 'cate_action_2_ratio', 
               'cate_action_3_ratio', 'cate_action_5_ratio', 'cate_action_6_ratio', 'cate_action_1_mean',
               'cate_action_2_mean', 'cate_action_3_mean', 'cate_action_4_mean', 'cate_action_5_mean',
               'cate_action_6_mean', 'cate_action_1_std', 'cate_action_2_std', 'cate_action_3_std',
               'cate_action_4_std', 'cate_action_5_std', 'cate_action_6_std']
    print('类别特征(get_accumulate_cate_feat):{}=>{}'.format(start_date,end_date))
    actions = get_actions(start_date, end_date, all_actions)
    actions['date'] = pd.to_datetime(actions['time']).apply(lambda x: x.date())
    df = pd.get_dummies(actions['type'], prefix='cate_action')
    actions = pd.concat([actions[['cate','date']], df], axis=1)
    # 按照类别-日期分组计算针对不同类别的各种行为某段时间的标准差
    actions_date = actions.groupby(['cate','date']).sum()
    actions_date = actions_date.unstack()
    actions_date.fillna(0, inplace=True)
    action_1 = np.std(actions_date['cate_action_1'], axis=1)
    action_1 = action_1.to_frame()
    action_1.columns = ['cate_action_1_std']
    action_2 = np.std(actions_date['cate_action_2'], axis=1)
    action_2 = action_2.to_frame()
    action_2.columns = ['cate_action_2_std']
    action_3 = np.std(actions_date['cate_action_3'], axis=1)
    action_3 = action_3.to_frame()
    action_3.columns = ['cate_action_3_std']
    action_4 = np.std(actions_date['cate_action_4'], axis=1)
    action_4 = action_4.to_frame()
    action_4.columns = ['cate_action_4_std']
    action_5 = np.std(actions_date['cate_action_5'], axis=1)
    action_5 = action_5.to_frame()
    action_5.columns = ['cate_action_5_std']
    action_6 = np.std(actions_date['cate_action_6'], axis=1)
    action_6 = action_6.to_frame()
    action_6.columns = ['cate_action_6_std']
    actions_date = pd.concat([action_1, action_2, action_3, action_4, action_5, action_6], axis=1)
    actions_date['cate'] = actions_date.index
    # 按照类别分组，统计各个商品类别下行为的转化率
    actions = actions.groupby(['cate'], as_index=False).sum()
    days_interal = (datetime.strptime(end_date, '%Y-%m-%d')-datetime.strptime(start_date, '%Y-%m-%d')).days
    
#     actions['cate_action_1_ratio'] = actions['cate_action_4'] / actions['cate_action_1']
#     actions['cate_action_2_ratio'] = actions['cate_action_4'] / actions['cate_action_2']
#     actions['cate_action_3_ratio'] = actions['cate_action_4'] / actions['cate_action_3']
#     actions['cate_action_5_ratio'] = actions['cate_action_4'] / actions['cate_action_5']
#     actions['cate_action_6_ratio'] = actions['cate_action_4'] / actions['cate_action_6']
    actions['cate_action_1_ratio'] =(np.log(1 + actions['cate_action_4']) - np.log(1 + actions['cate_action_1']))
    actions['cate_action_2_ratio'] =(np.log(1 + actions['cate_action_4']) - np.log(1 + actions['cate_action_2']))
    actions['cate_action_3_ratio'] =(np.log(1 + actions['cate_action_4']) - np.log(1 + actions['cate_action_3']))
    actions['cate_action_5_ratio'] =(np.log(1 + actions['cate_action_4']) - np.log(1 + actions['cate_action_5']))
    actions['cate_action_6_ratio'] =(np.log(1 + actions['cate_action_4']) - np.log(1 + actions['cate_action_6']))
    # 按照类别分组，统计各个商品类别下行为在一段时间的均值
    actions['cate_action_1_mean'] = actions['cate_action_1'] /  days_interal
    actions['cate_action_2_mean'] = actions['cate_action_2'] /  days_interal
    actions['cate_action_3_mean'] = actions['cate_action_3'] /  days_interal
    actions['cate_action_4_mean'] = actions['cate_action_4'] /  days_interal
    actions['cate_action_5_mean'] = actions['cate_action_5'] /  days_interal
    actions['cate_action_6_mean'] = actions['cate_action_6'] /  days_interal
    actions = pd.merge(actions, actions_date, how ='left',on='cate')
    actions = actions[feature]
    return actions


# In[28]:


get_accumulate_cate_feat('2016-02-01','2016-03-01',all_actions).info()


# ## 4.8. 构造训练集/验证集
# 
# * 标签,采用滑动窗口的方式，构造训练集的时候针对产生购买的行为标记为1
# * 整合特征
# 

# In[18]:


def get_labels(start_date,end_date,all_actions):
    print('label(get_labels):{}=>{}'.format(start_date,end_date))
    actions = get_actions(start_date,end_date,all_actions)
    #actions = actions[actions['type']==4]
    #修改为预测购买了商品8的用户预测
    actions = actions[(actions['type']==4) &(actions['cate']==8)]
    actions = actions.groupby(['user_id','sku_id'],as_index=False).sum()
    actions['label'] =1
    actions = actions[['user_id','sku_id','label']]
    return actions


# In[30]:


get_labels('2016-03-01','2016-04-01',all_actions).shape


# In[31]:


labels = get_labels('2016-04-04','2016-04-09',all_actions)
labels.drop_duplicates(subset=['user_id','sku_id']).shape


# 构造训练集

# In[35]:


# train_start_date = '2016-03-01'
def make_actions(user, product, all_actions, train_start_date):
    train_end_date = datetime.strptime(train_start_date, '%Y-%m-%d') + timedelta(days=3) #将时间往后推3天，得到截止时间
    train_end_date = train_end_date.strftime('%Y-%m-%d')
    # 修正prod_acc,cate_acc的时间跨度
    start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=30) #将截止时间往前移30天
    start_days = start_days.strftime('%Y-%m-%d')
    print('train_start_date:{},train_end_date:{},start_days:{}'.format(train_start_date,train_end_date,start_days))
    print ('train_end_date',train_end_date)
    user_acc = get_recent_user_feat(train_end_date, all_actions)
    print ('get_recent_user_feat finsihed')
    
    user_cate = get_user_cate_feature(train_start_date, train_end_date, all_actions)
    print ('get_user_cate_feature finished')
    
    product_acc = get_accumulate_product_feat(start_days, train_end_date, all_actions)
    print ('get_accumulate_product_feat finsihed')
    cate_acc = get_accumulate_cate_feat(start_days, train_end_date, all_actions)
    print ('get_accumulate_cate_feat finsihed')
    comment_acc = get_comments_product_feat(train_end_date)
    print ('get_comments_product_feat finished')
    # 标记
    test_start_date = train_end_date
    test_end_date = datetime.strptime(test_start_date, '%Y-%m-%d') + timedelta(days=5)#未来5天
    test_end_date = test_end_date.strftime('%Y-%m-%d')
#     labels = get_labels(test_start_date, test_end_date, all_actions)
    labels = get_labels('2016-03-01','2016-04-01',all_actions)# 由于数据问题，故固定日期
    print('len(labels)>>>>>>>>>>>>>>>>>>'+str(len(labels)))
    print ("get labels finished")
    
    actions = None
    for i in (3, 5, 7, 10, 15, 21, 30):
        start_days = datetime.strptime(train_end_date, 
                                       '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        if actions is None:
            actions = get_action_feat(start_days, 
                                      train_end_date, all_actions, i)
            print('get_action_feat,actions.shape:{}'.format(actions.shape))
        else:
            # 注意这里的拼接key
            data = get_action_feat(start_days,
                                   train_end_date,
                                   all_actions, i)
            print('get_action_feat,actions.shape:{}'.format(data.shape))
            actions = pd.merge(actions,data ,
                               how='left',
                               on=['user_id', 'sku_id', 'cate'])
    print('len(actions)================'+str(len(actions)))
    actions = pd.merge(actions, user, how='left', on='user_id')
    actions = pd.merge(actions, user_acc, how='left', on='user_id')
    actions = pd.merge(actions, user_cate, how='left', on='user_id')
    # 注意这里的拼接key
    actions = pd.merge(actions, product, how='left', on=['sku_id', 'cate'])
    actions = pd.merge(actions, product_acc, how='left', on='sku_id')
    actions = pd.merge(actions, cate_acc, how='left', on='cate')
    actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
    actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
    print('len(actions)2================'+str(len(actions)))
    # 主要是填充拼接商品基本特征、评论特征、标签之后的空值
    actions = actions.fillna(0)
#     return actions
    # 采样
    action_postive = actions[actions['label'] == 1]
    action_negative = actions[actions['label'] == 0]
    print('len(action_positive)'+str(len(action_postive)))
    print('len(action_negative)'+str(len(action_negative)))
    del actions
    neg_len = len(action_postive) * 10
    action_negative = action_negative.sample(n=neg_len)
    action_sample = pd.concat([action_postive, action_negative], ignore_index=True)
    print('len(action_sample)================'+str(len(action_sample)))
    
    return action_sample


# In[37]:


def make_train_set(train_start_date, setNums ,f_path):
    train_actions = None
    all_actions = get_all_action()
    print ("get all actions!")
    user = get_basic_user_feat()
    print ('get_basic_user_feat finsihed')
    product = get_basic_product_feat()
    print ('get_basic_product_feat finsihed')
    # 滑窗,构造多组训练集/验证集
    for i in range(setNums):
        print ('make_train_set方法的for循环中：',train_start_date)
        if train_actions is None:
            train_actions = make_actions(user, product, all_actions, train_start_date)
        else:
            train_actions = pd.concat([train_actions, make_actions(user, product, all_actions, train_start_date)],
                                          ignore_index=True)
        # 接下来每次移动一天
        train_start_date = datetime.strptime(train_start_date, '%Y-%m-%d') + timedelta(days=1)
        train_start_date = train_start_date.strftime('%Y-%m-%d')
        print ("round {0}/{1} over!".format(i+1, setNums))
#     train_actions.to_csv(f_path, index=False)
#     train_actions.iloc[:,0:10].to_csv('test.csv')


# In[34]:


# 训练集
# train_start_date = '2016-03-01'
# make_train_set(train_start_date, 2, 'train_set.csv')

train_start_date = '2016-04-01'
make_train_set(train_start_date, 2, 'train_set.csv')


# In[38]:


# 训练集
# train_start_date = '2016-03-01'
# make_train_set(train_start_date, 2, 'train_set.csv')

train_start_date = '2016-04-01'
make_train_set(train_start_date, 2, 'train_set.csv')


# 构造线下测试集

# In[35]:


label_val_s1_path='val_label.csv'
def make_val_answer(val_start_date, val_end_date, all_actions, label_val_s1_path):
    print('make_val_answer:{}=>{}'.format(val_start_date,val_end_date))
    actions = get_actions(val_start_date, val_end_date,all_actions)
    actions = actions[(actions['type'] == 4) & (actions['cate'] == 8)]
    actions = actions[['user_id', 'sku_id']]
    actions = actions.drop_duplicates()
    actions.to_csv(label_val_s1_path, index=False)


# In[36]:


make_val_answer('2016-03-01','2016-03-04',all_actions,label_val_s1_path)


# In[37]:


make_val_answer('2016-04-04','2016-04-09',all_actions,label_val_s1_path)


# In[38]:


def make_val_set(train_start_date, train_end_date, val_s1_path):
    # 修改时间跨度
    start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=30)
    start_days = start_days.strftime('%Y-%m-%d')
    print('make_val_set-------{}=>{},start_days:{}'.format(train_start_date,train_end_date,start_days))
    all_actions = get_all_action()
    print ("get all actions!")
    user = get_basic_user_feat()
    print ('get_basic_user_feat finsihed')
    
    product = get_basic_product_feat()
    print ('get_basic_product_feat finsihed')
#     user_acc = get_accumulate_user_feat(train_end_date,all_actions,30)
#     print 'get_accumulate_user_feat finished'
    user_acc = get_recent_user_feat(train_end_date, all_actions)
    print( 'get_recent_user_feat finsihed')
    user_cate = get_user_cate_feature(train_start_date, train_end_date, all_actions)
    print ('get_user_cate_feature finished')
 
    product_acc = get_accumulate_product_feat(start_days, train_end_date, all_actions)
    print ('get_accumulate_product_feat finsihed')
    cate_acc = get_accumulate_cate_feat(start_days, train_end_date, all_actions)
    print ('get_accumulate_cate_feat finsihed')
    comment_acc = get_comments_product_feat(train_end_date)
    print ('get_comments_product_feat finished')
    
    actions = None
    for i in (3, 5, 7, 10, 15, 21, 30):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        if actions is None:
            actions = get_action_feat(start_days, train_end_date, all_actions,i)
        else:
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date,all_actions,i), how='left',
                               on=['user_id', 'sku_id', 'cate'])
    actions = pd.merge(actions, user, how='left', on='user_id')
    actions = pd.merge(actions, user_acc, how='left', on='user_id')
    actions = pd.merge(actions, user_cate, how='left', on='user_id')
    # 注意这里的拼接key
    actions = pd.merge(actions, product, how='left', on=['sku_id', 'cate'])
    actions = pd.merge(actions, product_acc, how='left', on='sku_id')
    actions = pd.merge(actions, cate_acc, how='left', on='cate')
    actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
    actions = actions.fillna(0)
   
    
#     print actions
    # 构造真实用户购买情况作为后续验证
    val_start_date = train_end_date
    val_end_date = datetime.strptime(val_start_date, '%Y-%m-%d') + timedelta(days=5)
    val_end_date = val_end_date.strftime('%Y-%m-%d')
#     make_val_answer(val_start_date, val_end_date, all_actions, 'label_'+val_s1_path)
    make_val_answer('2016-03-01','2016-03-04',all_actions,'label_'+val_s1_path)
    
    actions.to_csv(val_s1_path, index=False)


# In[39]:


# 验证集
# train_start_date = '2016-04-06'
# make_train_set(train_start_date, 3, 'val_set.csv')
# make_val_set('2016-04-06', '2016-04-09', 'val_1.csv')
# make_val_set('2016-04-07', '2016-04-10', 'val_2.csv')
# make_val_set('2016-04-08', '2016-04-11', 'val_3.csv')

make_val_set('2016-04-01', '2016-04-04', 'val_1.csv')


# ## 4.9. 构造测试集

# In[40]:


def make_test_set(train_start_date, train_end_date):
    start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=30)
    start_days = start_days.strftime('%Y-%m-%d')
    print('make_test_set=========={}===>{},start_days:{}'.format(train_start_date,train_end_date,start_days))
    all_actions = get_all_action()
    print ("get all actions!")
    user = get_basic_user_feat()
    print ('get_basic_user_feat finsihed')
    product = get_basic_product_feat()
    print ('get_basic_product_feat finsihed')
    
    user_acc = get_recent_user_feat(train_end_date, all_actions)
    print ('get_accumulate_user_feat finsihed')
    
    user_cate = get_user_cate_feature(train_start_date, train_end_date, all_actions)
    print ('get_user_cate_feature finished')
    
    product_acc = get_accumulate_product_feat(start_days, train_end_date, all_actions)
    print ('get_accumulate_product_feat finsihed')
    cate_acc = get_accumulate_cate_feat(start_days, train_end_date, all_actions)
    print ('get_accumulate_cate_feat finsihed')
    comment_acc = get_comments_product_feat(train_end_date)
    actions = None
    for i in (3, 5, 7, 10, 15, 21, 30):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        if actions is None:
            actions = get_action_feat(start_days, train_end_date, all_actions,i)
        else:
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date,all_actions,i), how='left',
                               on=['user_id', 'sku_id', 'cate'])
    actions = pd.merge(actions, user, how='left', on='user_id')
    actions = pd.merge(actions, user_acc, how='left', on='user_id')
    actions = pd.merge(actions, user_cate, how='left', on='user_id')
    # 注意这里的拼接key
    actions = pd.merge(actions, product, how='left', on=['sku_id', 'cate'])
    actions = pd.merge(actions, product_acc, how='left', on='sku_id')
    actions = pd.merge(actions, cate_acc, how='left', on='cate')
    actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
    actions = actions.fillna(0)
    
    actions.to_csv("test_set.csv", index=False)


# 4.13~4.16这三天的评论记录似乎并不存在为0的情况，导致构建测试集时出错

# In[41]:


# # 预测结果
# sub_start_date = '2016-04-13'
# sub_end_date = '2016-04-16'
# make_test_set(sub_start_date, sub_end_date)

# 预测结果
sub_start_date = '2016-04-02'
sub_end_date = '2016-04-05'
make_test_set(sub_start_date, sub_end_date)


# # 5. 模型设计和评估

# In[42]:


import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import operator
from matplotlib import pylab as plt
from datetime import datetime
import time
from sklearn.model_selection import GridSearchCV


# In[43]:


def get_data_from_csv(FILE):
    reader = pd.read_csv(FILE,encoding='gbk',header=0,iterator=True)
    chunks=[]
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(1000000)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print('Iteration Stopped!!')
    df_ac = pd.concat(chunks, ignore_index=True)
    return df_ac


# In[44]:


get_data_from_csv('train_set.csv').head()


# In[45]:


def show_record():
    train = get_data_from_csv('train_set.csv')
#     valid = pd.read_csv('val_set.csv')
#     label_val = pd.read_csv('label_val_set.csv')
    valid1 = get_data_from_csv('val_1.csv')
#     valid2 = get_data_from_csv('val_2.csv')
#     valid3 = `get_data_from_csv('val_3.csv')
#     test = pd.read_csv('test_set.csv')
    print (train.shape)
#     print valid.shape
#     print label_val.shape
#     print test.shape
    print (valid1.shape)
#     print (valid2.shape)
#     print (valid3.shape)


# In[46]:


show_record()


# ## 5.1. 训练模型
# 
# * 返回训练后的模型
# * 生成特征map文件作为后续特征重要性之用
# 

# In[2]:


def create_feature_map(features):
    outfile = open(r'xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
    
def xgb_model(train_set):
    actions = get_data_from_csv(train_set)             #read train_set
    # 单纯的删掉模型前一遍训练认为无用的特征（根据特征重要性中不存在的特征）
    lst_useless = ['brand']
    
    actions.drop(lst_useless, inplace=True, axis=1)
    
    users = actions[['user_id', 'sku_id']].copy()
    labels = actions['label'].copy()
    del actions['user_id']
    del actions['sku_id']
    del actions['label']
    # 尝试通过设置scale_pos_weight来调整政府比例不均的问题，但是经过采样的正负比为1:10，
#     训练结果反而不如设置为1
#     ratio = float(np.sum(labels==0)) / np.sum(labels==1)
#     print ratio
    
    # write to feature map
    features = list(actions.columns[:])
    print ('total features: ', len(features))
    create_feature_map(features)
    # 训练时即传入特征名
#     features = list(actions.columns.values)
    
    user_index=users
    training_data=actions
    label=labels
    X_train, X_valid, y_train, y_valid = train_test_split(training_data.values, label, test_size=0.2, 
                                                          random_state=0)
    
    # 尝试通过提前设置传入训练的正负例的权重来改善正负比例不均的问题
#     weights = np.zeros(len(y_train))
#     weights[y_train==0] = 1
#     weights[y_train==1] = 10
    
#     dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
#     dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
#     dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=features)
#     dtrain = xgb.DMatrix(training_data.values, label.values)
    param = {'n_estimators': 4000, 'max_depth': 3, 'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 
             'colsample_bytree': 0.8, 'scale_pos_weight':10, 'eta': 0.1, 'silent': 1, 'objective': 'binary:logistic',
             'eval_metric':'auc'}
#     param = {'n_estimators': 4000, 'max_depth': 6, 'seed': 7, 'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 
#              'colsample_bytree': 0.8, 'scale_pos_weight': 1, 'eta': 0.09, 'silent': 1, 'objective': 'binary:logistic',
#              'eval_metric':'auc'}
    
    num_round = param['n_estimators']
#     param['nthread'] = 4
    # param['eval_metric'] = "auc"
    plst = param.items()
    evallist = [(dtrain, 'train'), (dvalid, 'eval')]
#     evallist = [(dvalid, 'eval'), (dtrain, 'train')]
#     evallist = [(dtrain, 'train')]
    bst = xgb.train(plst, dtrain, num_round, evallist, early_stopping_rounds=10)
    bst.save_model('bst.model')
    return bst, features

# bst_xgb, features = xgb_model('train_set.csv')


# In[76]:


bst_xgb, features = xgb_model('train_set.csv')


# ## 5.2. 对验证集进行线下测试

# In[77]:


def report(pred, label):
    actions = label
    result = pred
    # 所有实际用户商品对
    all_user_item_pair = actions['user_id'].map(str) + '-' + actions['sku_id'].map(str)
    all_user_item_pair = np.array(all_user_item_pair)
    # 所有购买用户
    all_user_set = actions['user_id'].unique()
    # 所有预测购买的用户
    all_user_test_set = result['user_id'].unique()
    all_user_test_item_pair = result['user_id'].map(str) + '-' + result['sku_id'].map(str)
    all_user_test_item_pair = np.array(all_user_test_item_pair)
    # 计算所有用户购买评价指标
    pos, neg = 0,0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
    all_user_acc = 1.0 * pos / ( pos + neg)
    all_user_recall = 1.0 * pos / len(all_user_set)
    print ('所有用户中预测购买用户的准确率为 ' + str(all_user_acc))
    print( '所有用户中预测购买用户的召回率' + str(all_user_recall))
    pos, neg = 0, 0
    for user_item_pair in all_user_test_item_pair:
        if user_item_pair in all_user_item_pair:
            pos += 1
        else:
            neg += 1
    all_item_acc = 1.0 * pos / ( pos + neg)
    all_item_recall = 1.0 * pos / len(all_user_item_pair)
    print( '所有用户中预测购买商品的准确率为 ' + str(all_item_acc))
    print ('所有用户中预测购买商品的召回率' + str(all_item_recall))
    F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
    F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc)
    score = 0.4 * F11 + 0.6 * F12
    print ('F11=' + str(F11))
    print ('F12=' + str(F12))
    print( 'score=' + str(score))
    
    return all_user_acc, all_user_recall, F11, all_item_acc, all_item_recall, F12, score


# In[84]:


def validate(valid_set, val_label, model):
    actions = get_data_from_csv(valid_set)                #read test_set        
#     users = actions[['user_id', 'sku_id']].copy()
    # 避免预测到非8类商品，所以最后还是再筛一遍的好        
    users = actions[['user_id', 'sku_id', 'cate']].copy()
    
    actions['user_id'] = actions['user_id'].astype(np.int64)
#     test_label= actions[actions['label'] == 1]
#     test_label= actions[(actions['label']==1) & (actions['cate']==8)]
    test_label = pd.read_csv(val_label)
    
    lst_useless = ['brand']
    
    actions.drop(lst_useless, inplace=True, axis=1)
    
#     test_label = test_label[['user_id','sku_id','label']]
    del actions['user_id']
    del actions['sku_id']
    print('len(actions.columns):==============='+str(len(actions.columns)))
    
#     features = list(actions.columns.values)
    
#     del actions['label']
    sub_user_index = users
#     sub_trainning_data = xgb.DMatrix(actions.values, feature_names=features)
    sub_trainning_data = xgb.DMatrix(actions.values)
#     y = model.predict(sub_trainning_data,ntree_limit=model.best_iteration)
    y = model.predict(sub_trainning_data, ntree_limit=model.best_ntree_limit)
    sub_user_index['label'] = y
    
    sub_user_index.to_csv('result_' + valid_set, index=False)
    
#     sub_user_index = sub_user_index[sub_user_index['cate']==8]
#     del sub_user_index['cate']
    rank = 1000
    pred = sub_user_index.sort_values(by='label', ascending=False)[:rank]
#     pred = sub_user_index[sub_user_index['label'] >= 0.05]
    
    print ('No. of raw pred users: ', len(pred['user_id'].unique()))
    pred = pred[pred['cate']==8]
    print ('No. of pred users bought cate 8: ', len(pred['user_id'].unique()))
    
#     pred = pred[['user_id', 'sku_id']]
    pred = pred[['user_id', 'sku_id', 'label']]
    pred = pred.groupby('user_id').first().reset_index()
    pred['user_id'] = pred['user_id'].astype(int)
    pred['sku_id'] = pred['sku_id'].astype(int)
    
#     print 'No. of pred users after deduplicates: ', len(pred['user_id'].unique())
    true_user = len(test_label['user_id'])   
    pred_ui = len(pred['user_id'].unique())
    print ('pred item: ', len(pred['sku_id'].unique()))
    print( 'true users: ', true_user)
    print ('pred users: ', pred_ui)
    test_label['user_id'] = test_label['user_id'].astype(int)
    test_label['sku_id'] = test_label['sku_id'].astype(int)
    all_user_acc, all_user_recall, F11, all_item_acc, all_item_recall, F12, score = report(pred, test_label)   
    
    f_name = 'pred_' + str(rank) + '_' + valid_set
    pred.to_csv(f_name, index=False)
    
    return rank, true_user, pred_ui, all_user_acc, all_user_recall, F11, all_item_acc, all_item_recall, F12, score
# validate('val_set.csv', bst_xgb)


# 评分文件

# In[85]:


def avg_score():
    rank1, true_user1, pred_ui1, user_acc1, user_recall1, F11_1, item_acc1,item_recall1, F12_1, score1 = validate('val_1.csv', 'label_val_1.csv', bst_xgb)
    print ('-------------------------------------------')
    rank2, true_user2, pred_ui2, user_acc2, user_recall2, F11_2, item_acc2,item_recall2, F12_2, score2 = validate('val_1.csv', 'label_val_1.csv', bst_xgb)
    print ('-------------------------------------------')
    rank3, true_user3, pred_ui3, user_acc3, user_recall3, F11_3, item_acc3,item_recall3, F12_3, score3 = validate('val_1.csv', 'label_val_1.csv', bst_xgb)
    print ('===========================================')
    
    print ('avg user acc: ', (user_acc1+user_acc2+user_acc3)/3)
    print ('avg user recall: ', (user_recall1+user_recall2+user_recall3)/3)
    print ('avg item acc: ', (item_acc1+item_acc2+item_acc3)/3)
    print ('avg item recall: ', (item_recall1+item_recall2+item_recall3)/3)
    print( 'avg F11: ', (F11_1+F11_2+F11_3)/3)
    print( 'avg F12: ', (F12_1+F12_2+F12_3)/3)
    print ('avg score: ', (score1+score2+score3)/3)
    # make the csv file
    dct_score = {}
    dct_score['rank'] = [rank1, rank2, rank3]
    dct_score['true_user'] = [true_user1, true_user2, true_user3]
    dct_score['pred_ui'] = [pred_ui1, pred_ui2, pred_ui3]
    dct_score['user_acc'] = [user_acc1, user_acc2, user_acc3]
    dct_score['user_recall'] = [user_recall1, user_recall2, user_recall3]
    dct_score['F11'] = [F11_1, F11_2, F11_3]
    dct_score['item_acc'] = [item_acc1, item_acc2, item_acc3]
    dct_score['item_recall'] = [item_recall1, item_recall2, item_recall3]
    dct_score['F12'] = [F12_1, F12_2, F12_3]
    dct_score['score'] = [score1, score2, score3]
    column_order = ['rank', 'true_user', 'pred_ui', 'user_acc', 'user_recall', 'item_acc', 'item_recall', 'F11', 'F12', 
                    'score']
    df_score = pd.DataFrame(dct_score)
    file_name = 'score_' + str(datetime.now().date())[5:] +'_'+ str(rank1) + '.csv'
    df_score[column_order].to_csv(file_name, index=False)

# avg_score()


# In[86]:


avg_score()


# ## 5.3. 输出特征重要性

# In[66]:


def feature_importance(bst_xgb):
    importance = bst_xgb.get_fscore(fmap=r'xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    file_name = 'feature_importance_' + str(datetime.now().date())[5:] + '.csv'
    df.to_csv(file_name)
    
# feature_importance(bst_xgb)


# In[67]:


feature_importance(bst_xgb)


# ## 5.4. 生成提交结果

# In[89]:


def submit(pred_set, model):
    actions = pd.read_csv(pred_set)                #read test_set
    
#     print 'total user before: ', len(actions['user_id'].unique())
#     potential = pd.read_csv('potential_user_04-28.csv')
#     lst_user = potential['user_id'].unique().tolist()
#     actions = actions[actions['user_id'].isin(lst_user)]
#     print 'total user after: ', len(actions['user_id'].unique())
    # 提前去掉部分特征
    lst_useless = ['brand']
    
    actions.drop(lst_useless, inplace=True, axis=1)
    
    users = actions[['user_id', 'sku_id', 'cate']].copy()
#     users = actions[['user_id', 'sku_id']].copy()
    
    actions['user_id'] = actions['user_id'].astype(np.int64)
    del actions['user_id']
    del actions['sku_id']
    sub_user_index = users
    #这个地方，要么传入df，要么传入ndarray数组，不能训练模型的时候传入df，
    #测试的时候传入ndarray,否则会报特征不匹配。
    sub_trainning_data = xgb.DMatrix(actions.values)
    y = model.predict(sub_trainning_data, ntree_limit=model.best_ntree_limit)
    sub_user_index['label'] = y
    
#     sub_user_index = sub_user_index[sub_user_index['cate']==8]
#     del sub_user_index['cate']
    rank = 1200
    pred = sub_user_index.sort_values(by='label', ascending=False)[:rank]
#     pred = sub_user_index[sub_user_index['label'] >= 0.05]
#     pred = pred[['user_id', 'sku_id', 'label']]
#     pred = pred[pred['label']>0.45]
    print ('No. of raw pred users: ', len(pred['user_id'].unique()))
    pred = pred[pred['cate']==8]
    print ('No. of pred users bought cate 8: ', len(pred['user_id'].unique()))
    pred = pred[['user_id', 'sku_id']]
#     print 
    pred = pred.groupby('user_id').first().reset_index()
    pred['user_id'] = pred['user_id'].astype(int)
    pred['sku_id'] = pred['sku_id'].astype(int)
    sub_file = 'submission_' + str(rank) + '_' + str(datetime.now().date())[5:] + '.csv'
#     sub_file = 'submission_detail_' + str(datetime.now().date())[5:] + '.csv'
    pred.to_csv(sub_file, index=False, index_label=False)  
    
# submit('test_set.csv', bst_xgb)


# In[90]:


submit('test_set.csv', bst_xgb)


# ## 5.5. 提交结果去重(可选)

# 除去最后临近三天的发生购买行为的用户商品对

# In[72]:


from datetime import datetime
def sub_improv(action, sub_file):
    # 获取4月最近三天的目标用户商品对
    action_4 = pd.read_csv(action)
    action_4['time'] = pd.to_datetime(action_4['time']).apply(lambda x: x.date())
    aim_date = [datetime.strptime(s, '%Y-%m-%d').date() for s in ['2016-04-09', '2016-04-10', '2016-04-11' ,
                                                                  '2016-04-12', '2016-04-13', '2016-04-14', 
                                                                  '2016-04-15']]
    aim_action = action_4[(action_4['type']==4) & (action_4['cate']==8) & (action_4['time'].isin(aim_date))]
    aim_ui = aim_action['user_id'].map(int).map(str) + '-' + aim_action['sku_id'].map(str)
    # 拼接提交数据的用户商品
    sub = pd.read_csv(sub_file)
    before = sub.shape[0]
    sub_ui = sub['user_id'].map(str) + '-' + sub['sku_id'].map(str)
    # 交集
    lst_aim = aim_ui.unique().tolist()
    lst_sub = sub_ui.unique().tolist()
    lst_common = [i for i in lst_aim if i in lst_sub]
    dct_ui = {i.split('-')[0]: i.split('-')[1] for i in lst_common}
    # 从提交结果除掉交集部分
    for k in dct_ui:
        sub.drop(sub[(sub['user_id']==int(k)) & (sub['sku_id']==int(dct_ui[k]))].index, inplace=True)
    print ('No. of records after remove dup: ', sub.shape[0])
    print ('No. of dup: ', before - sub.shape[0])
    if (before - sub.shape[0])!=0:
        file_name = 'submission_' + str(datetime.now().date())[5:] + '_improv.csv'
        sub.to_csv(file_name, index=False, index_label=False)
    
# sub_improv('data/JData_Action_201604.csv', 'submission_1200_05-25.csv')


# In[74]:


# sub_improv('data/JData_Action_201604.csv', 'submission_1200_05-25.csv')






