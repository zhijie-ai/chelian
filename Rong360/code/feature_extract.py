
#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/1/10 10:56                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
import seaborn as sns
p = sns.color_palette()
sns.set_style("darkgrid",{"font.sans-serif":['simhei', 'Arial']})

训练放款时间表 = pd.read_csv("../train/loan_time_train.txt",header=None,names=['用户标识','放款时间'])
训练放款时间表['放款时间']=训练放款时间表['放款时间']//86400

训练用户表 = pd.read_csv("../train/user_info_train.txt",header=None,
                    names=['用户标识','用户性别','用户职业','用户教育程度',
                           '用户婚姻状态', '用户户口类型'])
# 15个特征
训练信用卡账单表=pd.read_csv("../train/bill_detail_train.txt",header=None,
                    names=['用户标识','时间','银行标识','上期账单金额','上期还款金额','信用卡额度',
                           '本期账单余额','本期账单最低还款额','消费笔数','本期账单金额','调整金额',
                          '循环利息','可用余额','预借现金额度','还款状态'])
训练信用卡账单表['时间']=训练信用卡账单表['时间']//86400

训练信用卡账单表 = pd.merge(训练信用卡账单表, 训练放款时间表,how='inner', on = "用户标识")

用户浏览行为 = pd.read_csv("../train/browse_history_train.txt",header=None,
                    names=['用户标识','浏览时间','浏览行为数据','浏览子行为编号'])
用户浏览行为['浏览时间']=用户浏览行为['浏览时间']//86400
用户浏览行为= pd.merge(用户浏览行为, 训练放款时间表,how='inner', on = "用户标识")

银行流水记录=pd.read_csv("../train/bank_detail_train.txt").rename(index=str,
                                                         columns={"uid": "用户标识","timespan": "流水时间",
                                                                  "type":"交易类型","amount":"交易金额","markup":"工资收入标记"})
银行流水记录['流水时间']=银行流水记录['流水时间']//86400

训练表 = pd.read_csv("../train/overdue_train.txt",header=None,
                    names=['用户标识','标签'])

# 标签+用户信息+放款时间
训练表 = pd.merge(训练表,训练用户表,how='inner',on = "用户标识")
训练表 = pd.merge(训练表,训练放款时间表,how='inner',on = "用户标识")

训练表 = pd.read_csv("../train/overdue_train.txt",header=None,
                    names=['用户标识','标签'])

训练表 = pd.merge(训练表,训练用户表,how='inner',on = "用户标识")
训练表.to_csv("../feature/训练表_20170119_A.csv",index=None,encoding="gb2312")

# 训练放款时间表 = pd.read_csv("../train/loan_time_train.txt",header=None,names=['用户标识','放款时间'])
# 训练放款时间表['放款时间']=训练放款时间表['放款时间']//86400

测试放款时间表 = pd.read_csv("../test/loan_time_test.txt",header=None,names=['用户标识','放款时间'])
测试放款时间表['放款时间']=测试放款时间表['放款时间']//86400

# 训练信用卡账单表=pd.read_csv("../train/bill_detail_train.txt",header=None,
#                     names=['用户标识','时间','银行标识','上期账单金额','上期还款金额','信用卡额度',
#                            '本期账单余额','本期账单最低还款额','消费笔数','本期账单金额','调整金额',
#                           '循环利息','可用余额','预借现金额度','还款状态'])
# 训练信用卡账单表['时间']=训练信用卡账单表['时间']//86400
#
# 训练信用卡账单表 = pd.merge(训练信用卡账单表, 训练放款时间表,how='inner', on = "用户标识")

# 2  训练集特征构造:

# 3 补充特征：共55596 rows × 466 columns
# 放款前：
#train=pd.read_csv("../feature/train_20170116_A.csv",encoding="gb2312")
d=训练信用卡账单表[(训练信用卡账单表['时间']>0)]
feature=训练放款时间表
###补充特征，增加 放款前的各种统计信息sum count max min mean std var等
gb=d[(d['时间']<=d['放款时间'])].loc[:,['用户标识', '上期账单金额', '上期还款金额','信用卡额度','本期账单余额','本期账单最低还款额',
                                  '消费笔数','本期账单金额','调整金额','循环利息','可用余额','预借现金额度','还款状态']].groupby(["用户标识"],as_index=False)
放款前账单sum=gb.sum()
放款前账单sum.columns = ['用户标识', '放款前上期账单金额sum', '放款前上期还款金额sum','放款前信用卡额度sum','放款前本期账单余额sum',
                     '放款前本期账单最低还款额sum','放款前消费笔数sum','放款前本期账单金额sum','放款前调整金额sum','放款前循环利息sum',
                     '放款前可用余额sum','放款前预借现金额度sum','放款前还款状态sum']
feature=pd.merge(feature, 放款前账单sum,how='left', on = "用户标识")
feature['放款前上期还款金额sum与放款前上期账单金额sum差值']=feature['放款前上期还款金额sum']-feature['放款前上期账单金额sum']
feature['放款前信用卡额度sum与放款前本期账单余额sum差值']=feature['放款前信用卡额度sum']-feature['放款前本期账单余额sum']
feature['放款前可用余额sum与放款前预借现金额度sum差值']=feature['放款前可用余额sum']-feature['放款前预借现金额度sum']
feature['放款前本期账单最低还款额sum与放款前循环利息sum之和']=feature['放款前本期账单最低还款额sum']+feature['放款前循环利息sum']

放款前账单count=gb.count()
放款前账单count.columns = ['用户标识', '放款前上期账单金额count', '放款前上期还款金额count','放款前信用卡额度count','放款前本期账单余额count',
                     '放款前本期账单最低还款额count','放款前消费笔数count','放款前本期账单金额count','放款前调整金额count','放款前循环利息count',
                     '放款前可用余额count','放款前预借现金额度count','放款前还款状态count']
feature=pd.merge(feature, 放款前账单count,how='left', on = "用户标识")

放款前账单max=gb.max()
放款前账单max.columns = ['用户标识', '放款前上期账单金额max', '放款前上期还款金额max','放款前信用卡额度max','放款前本期账单余额max',
                     '放款前本期账单最低还款额max','放款前消费笔数max','放款前本期账单金额max','放款前调整金额max','放款前循环利息max',
                     '放款前可用余额max','放款前预借现金额度max','放款前还款状态max']
feature=pd.merge(feature, 放款前账单max,how='left', on = "用户标识")
feature['放款前上期还款金额max与放款前上期账单金额max差值']=feature['放款前上期还款金额max']-feature['放款前上期账单金额max']
feature['放款前信用卡额度max与放款前本期账单余额max差值']=feature['放款前信用卡额度max']-feature['放款前本期账单余额max']
feature['放款前可用余额max与放款前预借现金额度max差值']=feature['放款前可用余额max']-feature['放款前预借现金额度max']
feature['放款前本期账单最低还款额max与放款前循环利息max之和']=feature['放款前本期账单最低还款额max']+feature['放款前循环利息max']

放款前账单min=gb.min()
放款前账单min.columns = ['用户标识', '放款前上期账单金额min', '放款前上期还款金额min','放款前信用卡额度min','放款前本期账单余额min',
                     '放款前本期账单最低还款额min','放款前消费笔数min','放款前本期账单金额min','放款前调整金额min','放款前循环利息min',
                     '放款前可用余额min','放款前预借现金额度min','放款前还款状态min']
feature=pd.merge(feature, 放款前账单min,how='left', on = "用户标识")
feature['放款前上期还款金额min与放款前上期账单金额min差值']=feature['放款前上期还款金额min']-feature['放款前上期账单金额min']
feature['放款前信用卡额度min与放款前本期账单余额min差值']=feature['放款前信用卡额度min']-feature['放款前本期账单余额min']
feature['放款前可用余额min与放款前预借现金额度min差值']=feature['放款前可用余额min']-feature['放款前预借现金额度min']
feature['放款前本期账单最低还款额min与放款前循环利息min之和']=feature['放款前本期账单最低还款额min']+feature['放款前循环利息min']

放款前账单mean=gb.mean()
放款前账单mean.columns = ['用户标识', '放款前上期账单金额mean', '放款前上期还款金额mean','放款前信用卡额度mean','放款前本期账单余额mean',
                     '放款前本期账单最低还款额mean','放款前消费笔数mean','放款前本期账单金额mean','放款前调整金额mean','放款前循环利息mean',
                     '放款前可用余额mean','放款前预借现金额度mean','放款前还款状态mean']
feature=pd.merge(feature, 放款前账单mean,how='left', on = "用户标识")
feature['放款前上期还款金额mean与放款前上期账单金额mean差值']=feature['放款前上期还款金额mean']-feature['放款前上期账单金额mean']
feature['放款前信用卡额度mean与放款前本期账单余额mean差值']=feature['放款前信用卡额度mean']-feature['放款前本期账单余额mean']
feature['放款前可用余额mean与放款前预借现金额度mean差值']=feature['放款前可用余额mean']-feature['放款前预借现金额度mean']
feature['放款前本期账单最低还款额mean与放款前循环利息mean之和']=feature['放款前本期账单最低还款额mean']+feature['放款前循环利息mean']

放款前账单median=gb.median()
放款前账单median.columns = ['用户标识', '放款前上期账单金额median', '放款前上期还款金额median','放款前信用卡额度median','放款前本期账单余额median',
                  '放款前本期账单最低还款额median','放款前消费笔数median','放款前本期账单金额median','放款前调整金额median',
                  '放款前循环利息median','放款前可用余额median','放款前预借现金额度median','放款前还款状态median']
feature=pd.merge(feature, 放款前账单median,how='left', on = "用户标识")
feature['放款前上期还款金额median与放款前上期账单金额median差值']=feature['放款前上期还款金额median']-feature['放款前上期账单金额median']
feature['放款前信用卡额度median与放款前本期账单余额median差值']=feature['放款前信用卡额度median']-feature['放款前本期账单余额median']
feature['放款前可用余额median与放款前预借现金额度median差值']=feature['放款前可用余额median']-feature['放款前预借现金额度median']
feature['放款前本期账单最低还款额median与放款前循环利息median之和']=feature['放款前本期账单最低还款额median']+feature['放款前循环利息median']

放款前账单std=gb.std()
放款前账单std.columns = ['用户标识', '放款前上期账单金额std', '放款前上期还款金额std','放款前信用卡额度std','放款前本期账单余额std',
                     '放款前本期账单最低还款额std','放款前消费笔数std','放款前本期账单金额std','放款前调整金额std','放款前循环利息std',
                     '放款前可用余额std','放款前预借现金额度std','放款前还款状态std']
feature=pd.merge(feature, 放款前账单std,how='left', on = "用户标识")
feature['放款前上期还款金额std与放款前上期账单金额std差值']=feature['放款前上期还款金额std']-feature['放款前上期账单金额std']
feature['放款前信用卡额度std与放款前本期账单余额std差值']=feature['放款前信用卡额度std']-feature['放款前本期账单余额std']
feature['放款前可用余额std与放款前预借现金额度std差值']=feature['放款前可用余额std']-feature['放款前预借现金额度std']
feature['放款前本期账单最低还款额std与放款前循环利息std之和']=feature['放款前本期账单最低还款额std']+feature['放款前循环利息std']

放款前账单var=gb.var()
放款前账单var.columns = ['用户标识', '放款前上期账单金额var', '放款前上期还款金额var','放款前信用卡额度var','放款前本期账单余额var',
                     '放款前本期账单最低还款额var','放款前消费笔数var','放款前本期账单金额var','放款前调整金额var','放款前循环利息var',
                     '放款前可用余额var','放款前预借现金额度var','放款前还款状态var']
feature=pd.merge(feature, 放款前账单var,how='left', on = "用户标识")
feature['放款前上期还款金额var与放款前上期账单金额var差值']=feature['放款前上期还款金额var']-feature['放款前上期账单金额var']
feature['放款前信用卡额度var与放款前本期账单余额var差值']=feature['放款前信用卡额度var']-feature['放款前本期账单余额var']
feature['放款前可用余额var与放款前预借现金额度var差值']=feature['放款前可用余额var']-feature['放款前预借现金额度var']
feature['放款前本期账单最低还款额var与放款前循环利息var之和']=feature['放款前本期账单最低还款额var']+feature['放款前循环利息var']

#统计行和列的缺失值数
#feature1=feature.fillna(-1)
#print((feature1==-1).sum(axis=0))
#print((feature1==-1).sum(axis=1))
#feature#55596 rows × 118 columns
feature#55596 rows × 126 columns 2017.02.03

# 去重：
#train=pd.read_csv("../feature/train_20170116_A.csv",encoding="gb2312")
#放款前账单去重统计特征
#d=训练信用卡账单表[(训练信用卡账单表['时间']>0)]
feature_beifen=feature
#按用户标识\时间\银行标识汇总统计（去重）,每个用户在每个银行每天的一条唯一记录
data=d[(d['时间']<=d['放款时间'])].loc[:,['用户标识','时间','银行标识','上期账单金额', '上期还款金额','信用卡额度','本期账单余额'
                                  ,'本期账单最低还款额','消费笔数','本期账单金额','调整金额','循环利息','可用余额'
                                  ,'预借现金额度']].groupby(["用户标识","时间","银行标识"],as_index=False).max()

gb=data.loc[:,['用户标识', '上期账单金额', '上期还款金额','信用卡额度','本期账单余额','本期账单最低还款额',
               '消费笔数','本期账单金额','调整金额','循环利息','可用余额','预借现金额度']].groupby(["用户标识"],as_index=False)

去重后放款前账单sum=gb.sum()
去重后放款前账单sum.columns = ['用户标识', '去重后放款前上期账单金额sum', '去重后放款前上期还款金额sum','去重后放款前信用卡额度sum'
                       ,'去重后放款前本期账单余额sum','去重后放款前本期账单最低还款额sum','去重后放款前消费笔数sum'
                       ,'去重后放款前本期账单金额sum','去重后放款前调整金额sum','去重后放款前循环利息sum','去重后放款前可用余额sum'
                       ,'去重后放款前预借现金额度sum']
feature=pd.merge(feature, 去重后放款前账单sum,how='left', on = "用户标识")
feature['去重后放款前上期还款金额sum与放款前上期账单金额sum差值']=feature['去重后放款前上期还款金额sum']-feature['去重后放款前上期账单金额sum']
feature['去重后放款前信用卡额度sum与放款前本期账单余额sum差值']=feature['去重后放款前信用卡额度sum']-feature['去重后放款前本期账单余额sum']
feature['去重后放款前可用余额sum与放款前预借现金额度sum差值']=feature['去重后放款前可用余额sum']-feature['去重后放款前预借现金额度sum']
feature['去重后放款前本期账单最低还款额sum与放款前循环利息sum之和']=feature['去重后放款前本期账单最低还款额sum']+feature['去重后放款前循环利息sum']

去重后放款前账单count=gb.count()
去重后放款前账单count.columns = ['用户标识', '去重后放款前上期账单金额count', '去重后放款前上期还款金额count','去重后放款前信用卡额度count'
                         ,'去重后放款前本期账单余额count','去重后放款前本期账单最低还款额count','去重后放款前消费笔数count'
                         ,'去重后放款前本期账单金额count','去重后放款前调整金额count','去重后放款前循环利息count'
                         ,'去重后放款前可用余额count','去重后放款前预借现金额度count']
feature=pd.merge(feature, 去重后放款前账单count,how='left', on = "用户标识")

去重后放款前账单max=gb.max()
去重后放款前账单max.columns = ['用户标识', '去重后放款前上期账单金额max', '去重后放款前上期还款金额max','去重后放款前信用卡额度max'
                       ,'去重后放款前本期账单余额max','去重后放款前本期账单最低还款额max','去重后放款前消费笔数max'
                       ,'去重后放款前本期账单金额max','去重后放款前调整金额max','去重后放款前循环利息max'
                       ,'去重后放款前可用余额max','去重后放款前预借现金额度max']
feature=pd.merge(feature, 去重后放款前账单max,how='left', on = "用户标识")
feature['去重后放款前上期还款金额max与放款前上期账单金额max差值']=feature['去重后放款前上期还款金额max']-feature['去重后放款前上期账单金额max']
feature['去重后放款前信用卡额度max与放款前本期账单余额max差值']=feature['去重后放款前信用卡额度max']-feature['去重后放款前本期账单余额max']
feature['去重后放款前可用余额max与放款前预借现金额度max差值']=feature['去重后放款前可用余额max']-feature['去重后放款前预借现金额度max']
feature['去重后放款前本期账单最低还款额max与放款前循环利息max之和']=feature['去重后放款前本期账单最低还款额max']+feature['去重后放款前循环利息max']

去重后放款前账单min=gb.min()
去重后放款前账单min.columns = ['用户标识', '去重后放款前上期账单金额min', '去重后放款前上期还款金额min','去重后放款前信用卡额度min'
                       ,'去重后放款前本期账单余额min','去重后放款前本期账单最低还款额min','去重后放款前消费笔数min'
                       ,'去重后放款前本期账单金额min','去重后放款前调整金额min','去重后放款前循环利息min'
                       ,'去重后放款前可用余额min','去重后放款前预借现金额度min']
feature=pd.merge(feature, 去重后放款前账单min,how='left', on = "用户标识")
feature['去重后放款前上期还款金额min与放款前上期账单金额min差值']=feature['去重后放款前上期还款金额min']-feature['去重后放款前上期账单金额min']
feature['去重后放款前信用卡额度min与放款前本期账单余额min差值']=feature['去重后放款前信用卡额度min']-feature['去重后放款前本期账单余额min']
feature['去重后放款前可用余额min与放款前预借现金额度min差值']=feature['去重后放款前可用余额min']-feature['去重后放款前预借现金额度min']
feature['去重后放款前本期账单最低还款额min与放款前循环利息min之和']=feature['去重后放款前本期账单最低还款额min']+feature['去重后放款前循环利息min']

去重后放款前账单mean=gb.mean()
去重后放款前账单mean.columns = ['用户标识', '去重后放款前上期账单金额mean', '去重后放款前上期还款金额mean','去重后放款前信用卡额度mean'
                        ,'去重后放款前本期账单余额mean','去重后放款前本期账单最低还款额mean','去重后放款前消费笔数mean'
                        ,'去重后放款前本期账单金额mean','去重后放款前调整金额mean','去重后放款前循环利息mean'
                        ,'去重后放款前可用余额mean','去重后放款前预借现金额度mean']
feature=pd.merge(feature, 去重后放款前账单mean,how='left', on = "用户标识")
feature['去重后放款前上期还款金额mean与放款前上期账单金额mean差值']=feature['去重后放款前上期还款金额mean']-feature['去重后放款前上期账单金额mean']
feature['去重后放款前信用卡额度mean与放款前本期账单余额mean差值']=feature['去重后放款前信用卡额度mean']-feature['去重后放款前本期账单余额mean']
feature['去重后放款前可用余额mean与放款前预借现金额度mean差值']=feature['去重后放款前可用余额mean']-feature['去重后放款前预借现金额度mean']
feature['去重后放款前本期账单最低还款额mean与放款前循环利息mean之和']=feature['去重后放款前本期账单最低还款额mean']+feature['去重后放款前循环利息mean']

去重后放款前账单median=gb.median()
去重后放款前账单median.columns = ['用户标识', '去重后放款前上期账单金额median', '去重后放款前上期还款金额median'
                          ,'去重后放款前信用卡额度median','去重后放款前本期账单余额median','去重后放款前本期账单最低还款额median'
                          ,'去重后放款前消费笔数median','去重后放款前本期账单金额median','去重后放款前调整金额median'
                          ,'去重后放款前循环利息median','去重后放款前可用余额median','去重后放款前预借现金额度median']
feature=pd.merge(feature, 去重后放款前账单median,how='left', on = "用户标识")
feature['去重后放款前上期还款金额median与放款前上期账单金额median差值']=feature['去重后放款前上期还款金额median']-feature['去重后放款前上期账单金额median']
feature['去重后放款前信用卡额度median与放款前本期账单余额median差值']=feature['去重后放款前信用卡额度median']-feature['去重后放款前本期账单余额median']
feature['去重后放款前可用余额median与放款前预借现金额度median差值']=feature['去重后放款前可用余额median']-feature['去重后放款前预借现金额度median']
feature['去重后放款前本期账单最低还款额median与放款前循环利息median之和']=feature['去重后放款前本期账单最低还款额median']+feature['去重后放款前循环利息median']

去重后放款前账单std=gb.std()
去重后放款前账单std.columns = ['用户标识', '去重后放款前上期账单金额std', '去重后放款前上期还款金额std','去重后放款前信用卡额度std'
                       ,'去重后放款前本期账单余额std','去重后放款前本期账单最低还款额std','去重后放款前消费笔数std'
                       ,'去重后放款前本期账单金额std','去重后放款前调整金额std','去重后放款前循环利息std'
                       ,'去重后放款前可用余额std','去重后放款前预借现金额度std']
feature=pd.merge(feature, 去重后放款前账单std,how='left', on = "用户标识")
feature['去重后放款前上期还款金额std与放款前上期账单金额std差值']=feature['去重后放款前上期还款金额std']-feature['去重后放款前上期账单金额std']
feature['去重后放款前信用卡额度std与放款前本期账单余额std差值']=feature['去重后放款前信用卡额度std']-feature['去重后放款前本期账单余额std']
feature['去重后放款前可用余额std与放款前预借现金额度std差值']=feature['去重后放款前可用余额std']-feature['去重后放款前预借现金额度std']
feature['去重后放款前本期账单最低还款额std与放款前循环利息std之和']=feature['去重后放款前本期账单最低还款额std']+feature['去重后放款前循环利息std']

去重后放款前账单var=gb.var()
去重后放款前账单var.columns = ['用户标识', '去重后放款前上期账单金额var', '去重后放款前上期还款金额var','去重后放款前信用卡额度var'
                       ,'去重后放款前本期账单余额var','去重后放款前本期账单最低还款额var','去重后放款前消费笔数var'
                       ,'去重后放款前本期账单金额var','去重后放款前调整金额var','去重后放款前循环利息var','去重后放款前可用余额var','去重后放款前预借现金额度var']
feature=pd.merge(feature, 去重后放款前账单var,how='left', on = "用户标识")
feature['去重后放款前上期还款金额var与放款前上期账单金额var差值']=feature['去重后放款前上期还款金额var']-feature['去重后放款前上期账单金额var']
feature['去重后放款前信用卡额度var与放款前本期账单余额var差值']=feature['去重后放款前信用卡额度var']-feature['去重后放款前本期账单余额var']
feature['去重后放款前可用余额var与放款前预借现金额度var差值']=feature['去重后放款前可用余额var']-feature['去重后放款前预借现金额度var']
feature['去重后放款前本期账单最低还款额var与放款前循环利息var之和']=feature['去重后放款前本期账单最低还款额var']+feature['去重后放款前循环利息var']

feature#55596 rows × 242 columns

# 4  补充特征:
# 放款后：
#train=pd.read_csv("../feature/train_20170116_A.csv",encoding="gb2312")
#d=训练信用卡账单表[(训练信用卡账单表['时间']>0)]
feature_beifen=feature
###补充特征，增加 放款后的各种统计信息sum count max min mean std var等
gb=d[(d['时间']>d['放款时间'])].loc[:,['用户标识', '上期账单金额', '上期还款金额','信用卡额度','本期账单余额','本期账单最低还款额',
                                  '消费笔数','本期账单金额','调整金额','循环利息','可用余额','预借现金额度']].groupby(["用户标识"],as_index=False)

放款后账单sum=gb.sum()
放款后账单sum.columns = ['用户标识', '放款后上期账单金额sum', '放款后上期还款金额sum','放款后信用卡额度sum','放款后本期账单余额sum',
                     '放款后本期账单最低还款额sum','放款后消费笔数sum','放款后本期账单金额sum','放款后调整金额sum','放款后循环利息sum',
                     '放款后可用余额sum','放款后预借现金额度sum']

feature=pd.merge(feature, 放款后账单sum,how='left', on = "用户标识")
feature['放款后上期还款金额sum与放款后上期账单金额sum差值']=feature['放款后上期还款金额sum']-feature['放款后上期账单金额sum']
feature['放款后信用卡额度sum与放款后本期账单余额sum差值']=feature['放款后信用卡额度sum']-feature['放款后本期账单余额sum']
feature['放款后可用余额sum与放款后预借现金额度sum差值']=feature['放款后可用余额sum']-feature['放款后预借现金额度sum']
feature['放款后本期账单最低还款额sum与放款后循环利息sum之和']=feature['放款后本期账单最低还款额sum']+feature['放款后循环利息sum']

放款后账单count=gb.count()
放款后账单count.columns = ['用户标识', '放款后上期账单金额count', '放款后上期还款金额count','放款后信用卡额度count','放款后本期账单余额count',
                     '放款后本期账单最低还款额count','放款后消费笔数count','放款后本期账单金额count','放款后调整金额count','放款后循环利息count',
                     '放款后可用余额count','放款后预借现金额度count']
feature=pd.merge(feature, 放款后账单count,how='left', on = "用户标识")

放款后账单max=gb.max()
放款后账单max.columns = ['用户标识', '放款后上期账单金额max', '放款后上期还款金额max','放款后信用卡额度max','放款后本期账单余额max',
                     '放款后本期账单最低还款额max','放款后消费笔数max','放款后本期账单金额max','放款后调整金额max','放款后循环利息max',
                     '放款后可用余额max','放款后预借现金额度max']
feature=pd.merge(feature, 放款后账单max,how='left', on = "用户标识")
feature['放款后上期还款金额max与放款后上期账单金额max差值']=feature['放款后上期还款金额max']-feature['放款后上期账单金额max']
feature['放款后信用卡额度max与放款后本期账单余额max差值']=feature['放款后信用卡额度max']-feature['放款后本期账单余额max']
feature['放款后可用余额max与放款后预借现金额度max差值']=feature['放款后可用余额max']-feature['放款后预借现金额度max']
feature['放款后本期账单最低还款额max与放款后循环利息max之和']=feature['放款后本期账单最低还款额max']+feature['放款后循环利息max']

放款后账单min=gb.min()
放款后账单min.columns = ['用户标识', '放款后上期账单金额min', '放款后上期还款金额min','放款后信用卡额度min','放款后本期账单余额min',
                     '放款后本期账单最低还款额min','放款后消费笔数min','放款后本期账单金额min','放款后调整金额min','放款后循环利息min',
                     '放款后可用余额min','放款后预借现金额度min']
feature=pd.merge(feature, 放款后账单min,how='left', on = "用户标识")
feature['放款后上期还款金额min与放款后上期账单金额min差值']=feature['放款后上期还款金额min']-feature['放款后上期账单金额min']
feature['放款后信用卡额度min与放款后本期账单余额min差值']=feature['放款后信用卡额度min']-feature['放款后本期账单余额min']
feature['放款后可用余额min与放款后预借现金额度min差值']=feature['放款后可用余额min']-feature['放款后预借现金额度min']
feature['放款后本期账单最低还款额min与放款后循环利息min之和']=feature['放款后本期账单最低还款额min']+feature['放款后循环利息min']

放款后账单mean=gb.mean()
放款后账单mean.columns = ['用户标识', '放款后上期账单金额mean', '放款后上期还款金额mean','放款后信用卡额度mean','放款后本期账单余额mean',
                     '放款后本期账单最低还款额mean','放款后消费笔数mean','放款后本期账单金额mean','放款后调整金额mean','放款后循环利息mean',
                     '放款后可用余额mean','放款后预借现金额度mean']
feature=pd.merge(feature, 放款后账单mean,how='left', on = "用户标识")
feature['放款后上期还款金额mean与放款后上期账单金额mean差值']=feature['放款后上期还款金额mean']-feature['放款后上期账单金额mean']
feature['放款后信用卡额度mean与放款后本期账单余额mean差值']=feature['放款后信用卡额度mean']-feature['放款后本期账单余额mean']
feature['放款后可用余额mean与放款后预借现金额度mean差值']=feature['放款后可用余额mean']-feature['放款后预借现金额度mean']
feature['放款后本期账单最低还款额mean与放款后循环利息mean之和']=feature['放款后本期账单最低还款额mean']+feature['放款后循环利息mean']

放款后账单median=gb.median()
放款后账单median.columns = ['用户标识', '放款后上期账单金额median', '放款后上期还款金额median','放款后信用卡额度median','放款后本期账单余额median',
                  '放款后本期账单最低还款额median','放款后消费笔数median','放款后本期账单金额median','放款后调整金额median',
                  '放款后循环利息median','放款后可用余额median','放款后预借现金额度median']
feature=pd.merge(feature, 放款后账单median,how='left', on = "用户标识")
feature['放款后上期还款金额median与放款后上期账单金额median差值']=feature['放款后上期还款金额median']-feature['放款后上期账单金额median']
feature['放款后信用卡额度median与放款后本期账单余额median差值']=feature['放款后信用卡额度median']-feature['放款后本期账单余额median']
feature['放款后可用余额median与放款后预借现金额度median差值']=feature['放款后可用余额median']-feature['放款后预借现金额度median']
feature['放款后本期账单最低还款额median与放款后循环利息median之和']=feature['放款后本期账单最低还款额median']+feature['放款后循环利息median']

放款后账单std=gb.std()
放款后账单std.columns = ['用户标识', '放款后上期账单金额std', '放款后上期还款金额std','放款后信用卡额度std','放款后本期账单余额std',
                     '放款后本期账单最低还款额std','放款后消费笔数std','放款后本期账单金额std','放款后调整金额std','放款后循环利息std',
                     '放款后可用余额std','放款后预借现金额度std']
feature=pd.merge(feature, 放款后账单std,how='left', on = "用户标识")
feature['放款后上期还款金额std与放款后上期账单金额std差值']=feature['放款后上期还款金额std']-feature['放款后上期账单金额std']
feature['放款后信用卡额度std与放款后本期账单余额std差值']=feature['放款后信用卡额度std']-feature['放款后本期账单余额std']
feature['放款后可用余额std与放款后预借现金额度std差值']=feature['放款后可用余额std']-feature['放款后预借现金额度std']
feature['放款后本期账单最低还款额std与放款后循环利息std之和']=feature['放款后本期账单最低还款额std']+feature['放款后循环利息std']

放款后账单var=gb.var()
放款后账单var.columns = ['用户标识', '放款后上期账单金额var', '放款后上期还款金额var','放款后信用卡额度var','放款后本期账单余额var',
                     '放款后本期账单最低还款额var','放款后消费笔数var','放款后本期账单金额var','放款后调整金额var','放款后循环利息var',
                     '放款后可用余额var','放款后预借现金额度var']
feature=pd.merge(feature, 放款后账单var,how='left', on = "用户标识")
feature['放款后上期还款金额var与放款后上期账单金额var差值']=feature['放款后上期还款金额var']-feature['放款后上期账单金额var']
feature['放款后信用卡额度var与放款后本期账单余额var差值']=feature['放款后信用卡额度var']-feature['放款后本期账单余额var']
feature['放款后可用余额var与放款后预借现金额度var差值']=feature['放款后可用余额var']-feature['放款后预借现金额度var']
feature['放款后本期账单最低还款额var与放款后循环利息var之和']=feature['放款后本期账单最低还款额var']+feature['放款后循环利息var']

feature#55596 rows × 358 columns

#train=pd.read_csv("../feature/train_20170116_A.csv",encoding="gb2312")
#放款后账单去重统计特征
#d=训练信用卡账单表[(训练信用卡账单表['时间']>0)]
feature_beifen=feature
#按用户标识\时间\银行标识汇总统计（去重）
data=d[(d['时间']>d['放款时间'])].loc[:,['用户标识','时间','银行标识','上期账单金额', '上期还款金额','信用卡额度','本期账单余额'
                                  ,'本期账单最低还款额','消费笔数','本期账单金额','调整金额','循环利息','可用余额'
                                  ,'预借现金额度']].groupby(["用户标识","时间","银行标识"],as_index=False).max()
gb=data.loc[:,['用户标识', '上期账单金额', '上期还款金额','信用卡额度','本期账单余额','本期账单最低还款额',
               '消费笔数','本期账单金额','调整金额','循环利息','可用余额','预借现金额度']].groupby(["用户标识"],as_index=False)

去重后放款后账单sum=gb.sum()
去重后放款后账单sum.columns = ['用户标识', '去重后放款后上期账单金额sum', '去重后放款后上期还款金额sum','去重后放款后信用卡额度sum'
                       ,'去重后放款后本期账单余额sum','去重后放款后本期账单最低还款额sum','去重后放款后消费笔数sum'
                       ,'去重后放款后本期账单金额sum','去重后放款后调整金额sum','去重后放款后循环利息sum','去重后放款后可用余额sum'
                       ,'去重后放款后预借现金额度sum']
feature=pd.merge(feature, 去重后放款后账单sum,how='left', on = "用户标识")
feature['去重后放款后上期还款金额sum与放款后上期账单金额sum差值']=feature['去重后放款后上期还款金额sum']-feature['去重后放款后上期账单金额sum']
feature['去重后放款后信用卡额度sum与放款后本期账单余额sum差值']=feature['去重后放款后信用卡额度sum']-feature['去重后放款后本期账单余额sum']
feature['去重后放款后可用余额sum与放款后预借现金额度sum差值']=feature['去重后放款后可用余额sum']-feature['去重后放款后预借现金额度sum']
feature['去重后放款后本期账单最低还款额sum与放款后循环利息sum之和']=feature['去重后放款后本期账单最低还款额sum']+feature['去重后放款后循环利息sum']

去重后放款后账单count=gb.count()
去重后放款后账单count.columns = ['用户标识', '去重后放款后上期账单金额count', '去重后放款后上期还款金额count','去重后放款后信用卡额度count'
                         ,'去重后放款后本期账单余额count','去重后放款后本期账单最低还款额count','去重后放款后消费笔数count'
                         ,'去重后放款后本期账单金额count','去重后放款后调整金额count','去重后放款后循环利息count'
                         ,'去重后放款后可用余额count','去重后放款后预借现金额度count']
feature=pd.merge(feature, 去重后放款后账单count,how='left', on = "用户标识")

去重后放款后账单max=gb.max()
去重后放款后账单max.columns = ['用户标识', '去重后放款后上期账单金额max', '去重后放款后上期还款金额max','去重后放款后信用卡额度max'
                       ,'去重后放款后本期账单余额max','去重后放款后本期账单最低还款额max','去重后放款后消费笔数max'
                       ,'去重后放款后本期账单金额max','去重后放款后调整金额max','去重后放款后循环利息max'
                       ,'去重后放款后可用余额max','去重后放款后预借现金额度max']
feature=pd.merge(feature, 去重后放款后账单max,how='left', on = "用户标识")
feature['去重后放款后上期还款金额max与放款后上期账单金额max差值']=feature['去重后放款后上期还款金额max']-feature['去重后放款后上期账单金额max']
feature['去重后放款后信用卡额度max与放款后本期账单余额max差值']=feature['去重后放款后信用卡额度max']-feature['去重后放款后本期账单余额max']
feature['去重后放款后可用余额max与放款后预借现金额度max差值']=feature['去重后放款后可用余额max']-feature['去重后放款后预借现金额度max']
feature['去重后放款后本期账单最低还款额max与放款后循环利息max之和']=feature['去重后放款后本期账单最低还款额max']+feature['去重后放款后循环利息max']

去重后放款后账单min=gb.min()
去重后放款后账单min.columns = ['用户标识', '去重后放款后上期账单金额min', '去重后放款后上期还款金额min','去重后放款后信用卡额度min'
                       ,'去重后放款后本期账单余额min','去重后放款后本期账单最低还款额min','去重后放款后消费笔数min'
                       ,'去重后放款后本期账单金额min','去重后放款后调整金额min','去重后放款后循环利息min'
                       ,'去重后放款后可用余额min','去重后放款后预借现金额度min']
feature=pd.merge(feature, 去重后放款后账单min,how='left', on = "用户标识")
feature['去重后放款后上期还款金额min与放款后上期账单金额min差值']=feature['去重后放款后上期还款金额min']-feature['去重后放款后上期账单金额min']
feature['去重后放款后信用卡额度min与放款后本期账单余额min差值']=feature['去重后放款后信用卡额度min']-feature['去重后放款后本期账单余额min']
feature['去重后放款后可用余额min与放款后预借现金额度min差值']=feature['去重后放款后可用余额min']-feature['去重后放款后预借现金额度min']
feature['去重后放款后本期账单最低还款额min与放款后循环利息min之和']=feature['去重后放款后本期账单最低还款额min']+feature['去重后放款后循环利息min']

去重后放款后账单mean=gb.mean()
去重后放款后账单mean.columns = ['用户标识', '去重后放款后上期账单金额mean', '去重后放款后上期还款金额mean','去重后放款后信用卡额度mean'
                        ,'去重后放款后本期账单余额mean','去重后放款后本期账单最低还款额mean','去重后放款后消费笔数mean'
                        ,'去重后放款后本期账单金额mean','去重后放款后调整金额mean','去重后放款后循环利息mean'
                        ,'去重后放款后可用余额mean','去重后放款后预借现金额度mean']
feature=pd.merge(feature, 去重后放款后账单mean,how='left', on = "用户标识")
feature['去重后放款后上期还款金额mean与放款后上期账单金额mean差值']=feature['去重后放款后上期还款金额mean']-feature['去重后放款后上期账单金额mean']
feature['去重后放款后信用卡额度mean与放款后本期账单余额mean差值']=feature['去重后放款后信用卡额度mean']-feature['去重后放款后本期账单余额mean']
feature['去重后放款后可用余额mean与放款后预借现金额度mean差值']=feature['去重后放款后可用余额mean']-feature['去重后放款后预借现金额度mean']
feature['去重后放款后本期账单最低还款额mean与放款后循环利息mean之和']=feature['去重后放款后本期账单最低还款额mean']+feature['去重后放款后循环利息mean']


去重后放款后账单median=gb.median()
去重后放款后账单median.columns = ['用户标识', '去重后放款后上期账单金额median', '去重后放款后上期还款金额median'
                          ,'去重后放款后信用卡额度median','去重后放款后本期账单余额median','去重后放款后本期账单最低还款额median'
                          ,'去重后放款后消费笔数median','去重后放款后本期账单金额median','去重后放款后调整金额median'
                          ,'去重后放款后循环利息median','去重后放款后可用余额median','去重后放款后预借现金额度median']
feature=pd.merge(feature, 去重后放款后账单median,how='left', on = "用户标识")
feature['去重后放款后上期还款金额median与放款后上期账单金额median差值']=feature['去重后放款后上期还款金额median']-feature['去重后放款后上期账单金额median']
feature['去重后放款后信用卡额度median与放款后本期账单余额median差值']=feature['去重后放款后信用卡额度median']-feature['去重后放款后本期账单余额median']
feature['去重后放款后可用余额median与放款后预借现金额度median差值']=feature['去重后放款后可用余额median']-feature['去重后放款后预借现金额度median']
feature['去重后放款后本期账单最低还款额median与放款后循环利息median之和']=feature['去重后放款后本期账单最低还款额median']+feature['去重后放款后循环利息median']


去重后放款后账单std=gb.std()
去重后放款后账单std.columns = ['用户标识', '去重后放款后上期账单金额std', '去重后放款后上期还款金额std','去重后放款后信用卡额度std'
                       ,'去重后放款后本期账单余额std','去重后放款后本期账单最低还款额std','去重后放款后消费笔数std'
                       ,'去重后放款后本期账单金额std','去重后放款后调整金额std','去重后放款后循环利息std'
                       ,'去重后放款后可用余额std','去重后放款后预借现金额度std']
feature=pd.merge(feature, 去重后放款后账单std,how='left', on = "用户标识")
feature['去重后放款后上期还款金额std与放款后上期账单金额std差值']=feature['去重后放款后上期还款金额std']-feature['去重后放款后上期账单金额std']
feature['去重后放款后信用卡额度std与放款后本期账单余额std差值']=feature['去重后放款后信用卡额度std']-feature['去重后放款后本期账单余额std']
feature['去重后放款后可用余额std与放款后预借现金额度std差值']=feature['去重后放款后可用余额std']-feature['去重后放款后预借现金额度std']
feature['去重后放款后本期账单最低还款额std与放款后循环利息std之和']=feature['去重后放款后本期账单最低还款额std']+feature['去重后放款后循环利息std']

去重后放款后账单var=gb.var()
去重后放款后账单var.columns = ['用户标识', '去重后放款后上期账单金额var', '去重后放款后上期还款金额var','去重后放款后信用卡额度var'
                       ,'去重后放款后本期账单余额var','去重后放款后本期账单最低还款额var','去重后放款后消费笔数var'
                       ,'去重后放款后本期账单金额var','去重后放款后调整金额var','去重后放款后循环利息var','去重后放款后可用余额var','去重后放款后预借现金额度var']
feature=pd.merge(feature, 去重后放款后账单var,how='left', on = "用户标识")
feature['去重后放款后上期还款金额var与放款后上期账单金额var差值']=feature['去重后放款后上期还款金额var']-feature['去重后放款后上期账单金额var']
feature['去重后放款后信用卡额度var与放款后本期账单余额var差值']=feature['去重后放款后信用卡额度var']-feature['去重后放款后本期账单余额var']
feature['去重后放款后可用余额var与放款后预借现金额度var差值']=feature['去重后放款后可用余额var']-feature['去重后放款后预借现金额度var']
feature['去重后放款后本期账单最低还款额var与放款后循环利息var之和']=feature['去重后放款后本期账单最低还款额var']+feature['去重后放款后循环利息var']

feature#55596 rows × 474 columns
feature.to_csv("../feature/用户账单表特征训练表20170203_时间已知.csv",index=None,encoding="gb2312")#55596 rows × 474 columns

# 时间未知：
#train=pd.read_csv("../feature/train_20170116_A.csv",encoding="gb2312")
d=训练信用卡账单表[(训练信用卡账单表['时间']==0)]
feature=训练放款时间表[['用户标识']]
###补充特征，增加 时间未知的各种统计信息sum count max min mean std var等
gb=d.loc[:,['用户标识', '上期账单金额', '上期还款金额','信用卡额度','本期账单余额','本期账单最低还款额',
                                  '消费笔数','本期账单金额','调整金额','循环利息','可用余额','预借现金额度','还款状态']].groupby(["用户标识"],as_index=False)

时间未知账单sum=gb.sum()
时间未知账单sum.columns = ['用户标识', '时间未知上期账单金额sum', '时间未知上期还款金额sum','时间未知信用卡额度sum','时间未知本期账单余额sum',
                     '时间未知本期账单最低还款额sum','时间未知消费笔数sum','时间未知本期账单金额sum','时间未知调整金额sum','时间未知循环利息sum',
                     '时间未知可用余额sum','时间未知预借现金额度sum','时间未知还款状态sum']
feature=pd.merge(feature, 时间未知账单sum,how='left', on = "用户标识")
feature['时间未知上期还款金额sum与时间未知上期账单金额sum差值']=feature['时间未知上期还款金额sum']-feature['时间未知上期账单金额sum']
feature['时间未知信用卡额度sum与时间未知本期账单余额sum差值']=feature['时间未知信用卡额度sum']-feature['时间未知本期账单余额sum']
feature['时间未知可用余额sum与时间未知预借现金额度sum差值']=feature['时间未知可用余额sum']-feature['时间未知预借现金额度sum']
feature['时间未知本期账单最低还款额sum与时间未知循环利息sum之和']=feature['时间未知本期账单最低还款额sum']+feature['时间未知循环利息sum']

时间未知账单count=gb.count()
时间未知账单count.columns = ['用户标识', '时间未知上期账单金额count', '时间未知上期还款金额count','时间未知信用卡额度count','时间未知本期账单余额count',
                     '时间未知本期账单最低还款额count','时间未知消费笔数count','时间未知本期账单金额count','时间未知调整金额count','时间未知循环利息count',
                     '时间未知可用余额count','时间未知预借现金额度count','时间未知还款状态count']
feature=pd.merge(feature, 时间未知账单count,how='left', on = "用户标识")

时间未知账单max=gb.max()
时间未知账单max.columns = ['用户标识', '时间未知上期账单金额max', '时间未知上期还款金额max','时间未知信用卡额度max','时间未知本期账单余额max',
                     '时间未知本期账单最低还款额max','时间未知消费笔数max','时间未知本期账单金额max','时间未知调整金额max','时间未知循环利息max',
                     '时间未知可用余额max','时间未知预借现金额度max','时间未知还款状态max']
feature=pd.merge(feature, 时间未知账单max,how='left', on = "用户标识")
feature['时间未知上期还款金额max与时间未知上期账单金额max差值']=feature['时间未知上期还款金额max']-feature['时间未知上期账单金额max']
feature['时间未知信用卡额度max与时间未知本期账单余额max差值']=feature['时间未知信用卡额度max']-feature['时间未知本期账单余额max']
feature['时间未知可用余额max与时间未知预借现金额度max差值']=feature['时间未知可用余额max']-feature['时间未知预借现金额度max']
feature['时间未知本期账单最低还款额max与时间未知循环利息max之和']=feature['时间未知本期账单最低还款额max']+feature['时间未知循环利息max']

时间未知账单min=gb.min()
时间未知账单min.columns = ['用户标识', '时间未知上期账单金额min', '时间未知上期还款金额min','时间未知信用卡额度min','时间未知本期账单余额min',
                     '时间未知本期账单最低还款额min','时间未知消费笔数min','时间未知本期账单金额min','时间未知调整金额min','时间未知循环利息min',
                     '时间未知可用余额min','时间未知预借现金额度min','时间未知还款状态min']
feature=pd.merge(feature, 时间未知账单min,how='left', on = "用户标识")
feature['时间未知上期还款金额min与时间未知上期账单金额min差值']=feature['时间未知上期还款金额min']-feature['时间未知上期账单金额min']
feature['时间未知信用卡额度min与时间未知本期账单余额min差值']=feature['时间未知信用卡额度min']-feature['时间未知本期账单余额min']
feature['时间未知可用余额min与时间未知预借现金额度min差值']=feature['时间未知可用余额min']-feature['时间未知预借现金额度min']
feature['时间未知本期账单最低还款额min与时间未知循环利息min之和']=feature['时间未知本期账单最低还款额min']+feature['时间未知循环利息min']

时间未知账单mean=gb.mean()
时间未知账单mean.columns = ['用户标识', '时间未知上期账单金额mean', '时间未知上期还款金额mean','时间未知信用卡额度mean','时间未知本期账单余额mean',
                     '时间未知本期账单最低还款额mean','时间未知消费笔数mean','时间未知本期账单金额mean','时间未知调整金额mean','时间未知循环利息mean',
                     '时间未知可用余额mean','时间未知预借现金额度mean','时间未知还款状态mean']
feature=pd.merge(feature, 时间未知账单mean,how='left', on = "用户标识")
feature['时间未知上期还款金额mean与时间未知上期账单金额mean差值']=feature['时间未知上期还款金额mean']-feature['时间未知上期账单金额mean']
feature['时间未知信用卡额度mean与时间未知本期账单余额mean差值']=feature['时间未知信用卡额度mean']-feature['时间未知本期账单余额mean']
feature['时间未知可用余额mean与时间未知预借现金额度mean差值']=feature['时间未知可用余额mean']-feature['时间未知预借现金额度mean']
feature['时间未知本期账单最低还款额mean与时间未知循环利息mean之和']=feature['时间未知本期账单最低还款额mean']+feature['时间未知循环利息mean']

时间未知账单median=gb.median()
时间未知账单median.columns = ['用户标识', '时间未知上期账单金额median', '时间未知上期还款金额median','时间未知信用卡额度median','时间未知本期账单余额median',
                  '时间未知本期账单最低还款额median','时间未知消费笔数median','时间未知本期账单金额median','时间未知调整金额median',
                  '时间未知循环利息median','时间未知可用余额median','时间未知预借现金额度median','时间未知还款状态median']
feature=pd.merge(feature, 时间未知账单median,how='left', on = "用户标识")
feature['时间未知上期还款金额median与时间未知上期账单金额median差值']=feature['时间未知上期还款金额median']-feature['时间未知上期账单金额median']
feature['时间未知信用卡额度median与时间未知本期账单余额median差值']=feature['时间未知信用卡额度median']-feature['时间未知本期账单余额median']
feature['时间未知可用余额median与时间未知预借现金额度median差值']=feature['时间未知可用余额median']-feature['时间未知预借现金额度median']
feature['时间未知本期账单最低还款额median与时间未知循环利息median之和']=feature['时间未知本期账单最低还款额median']+feature['时间未知循环利息median']

时间未知账单std=gb.std()
时间未知账单std.columns = ['用户标识', '时间未知上期账单金额std', '时间未知上期还款金额std','时间未知信用卡额度std','时间未知本期账单余额std',
                     '时间未知本期账单最低还款额std','时间未知消费笔数std','时间未知本期账单金额std','时间未知调整金额std','时间未知循环利息std',
                     '时间未知可用余额std','时间未知预借现金额度std','时间未知还款状态std']
feature=pd.merge(feature, 时间未知账单std,how='left', on = "用户标识")
feature['时间未知上期还款金额std与时间未知上期账单金额std差值']=feature['时间未知上期还款金额std']-feature['时间未知上期账单金额std']
feature['时间未知信用卡额度std与时间未知本期账单余额std差值']=feature['时间未知信用卡额度std']-feature['时间未知本期账单余额std']
feature['时间未知可用余额std与时间未知预借现金额度std差值']=feature['时间未知可用余额std']-feature['时间未知预借现金额度std']
feature['时间未知本期账单最低还款额std与时间未知循环利息std之和']=feature['时间未知本期账单最低还款额std']+feature['时间未知循环利息std']

时间未知账单var=gb.var()
时间未知账单var.columns = ['用户标识', '时间未知上期账单金额var', '时间未知上期还款金额var','时间未知信用卡额度var','时间未知本期账单余额var',
                     '时间未知本期账单最低还款额var','时间未知消费笔数var','时间未知本期账单金额var','时间未知调整金额var','时间未知循环利息var',
                     '时间未知可用余额var','时间未知预借现金额度var','时间未知还款状态var']
feature=pd.merge(feature, 时间未知账单var,how='left', on = "用户标识")
feature['时间未知上期还款金额var与时间未知上期账单金额var差值']=feature['时间未知上期还款金额var']-feature['时间未知上期账单金额var']
feature['时间未知信用卡额度var与时间未知本期账单余额var差值']=feature['时间未知信用卡额度var']-feature['时间未知本期账单余额var']
feature['时间未知可用余额var与时间未知预借现金额度var差值']=feature['时间未知可用余额var']-feature['时间未知预借现金额度var']
feature['时间未知本期账单最低还款额var与时间未知循环利息var之和']=feature['时间未知本期账单最低还款额var']+feature['时间未知循环利息var']
feature#55596 rows × 125 columns

# 去重（时间未知）：
#train=pd.read_csv("../feature/train_20170116_A.csv",encoding="gb2312")
#时间未知账单去重统计特征
#d=训练信用卡账单表[(训练信用卡账单表['时间']==0)]
feature_beifen=feature
#按用户标识\时间\银行标识汇总统计（去重）
data=d.loc[:,['用户标识','时间','银行标识','上期账单金额', '上期还款金额','信用卡额度','本期账单余额'
              ,'本期账单最低还款额','消费笔数','本期账单金额','调整金额','循环利息','可用余额'
              ,'预借现金额度']].groupby(["用户标识","时间","银行标识"],as_index=False).max()

gb=data.loc[:,['用户标识', '上期账单金额', '上期还款金额','信用卡额度','本期账单余额','本期账单最低还款额',
               '消费笔数','本期账单金额','调整金额','循环利息','可用余额','预借现金额度']].groupby(["用户标识"],as_index=False)

去重后时间未知账单sum=gb.sum()
去重后时间未知账单sum.columns = ['用户标识', '去重后时间未知上期账单金额sum', '去重后时间未知上期还款金额sum','去重后时间未知信用卡额度sum'
                       ,'去重后时间未知本期账单余额sum','去重后时间未知本期账单最低还款额sum','去重后时间未知消费笔数sum'
                       ,'去重后时间未知本期账单金额sum','去重后时间未知调整金额sum','去重后时间未知循环利息sum','去重后时间未知可用余额sum'
                       ,'去重后时间未知预借现金额度sum']

feature=pd.merge(feature, 去重后时间未知账单sum,how='left', on = "用户标识")
feature['去重后时间未知上期还款金额sum与时间未知上期账单金额sum差值']=feature['去重后时间未知上期还款金额sum']-feature['去重后时间未知上期账单金额sum']
feature['去重后时间未知信用卡额度sum与时间未知本期账单余额sum差值']=feature['去重后时间未知信用卡额度sum']-feature['去重后时间未知本期账单余额sum']
feature['去重后时间未知可用余额sum与时间未知预借现金额度sum差值']=feature['去重后时间未知可用余额sum']-feature['去重后时间未知预借现金额度sum']
feature['去重后时间未知本期账单最低还款额sum与时间未知循环利息sum之和']=feature['去重后时间未知本期账单最低还款额sum']+feature['去重后时间未知循环利息sum']

去重后时间未知账单count=gb.count()
去重后时间未知账单count.columns = ['用户标识', '去重后时间未知上期账单金额count', '去重后时间未知上期还款金额count','去重后时间未知信用卡额度count'
                         ,'去重后时间未知本期账单余额count','去重后时间未知本期账单最低还款额count','去重后时间未知消费笔数count'
                         ,'去重后时间未知本期账单金额count','去重后时间未知调整金额count','去重后时间未知循环利息count'
                         ,'去重后时间未知可用余额count','去重后时间未知预借现金额度count']
feature=pd.merge(feature, 去重后时间未知账单count,how='left', on = "用户标识")

去重后时间未知账单max=gb.max()
去重后时间未知账单max.columns = ['用户标识', '去重后时间未知上期账单金额max', '去重后时间未知上期还款金额max','去重后时间未知信用卡额度max'
                       ,'去重后时间未知本期账单余额max','去重后时间未知本期账单最低还款额max','去重后时间未知消费笔数max'
                       ,'去重后时间未知本期账单金额max','去重后时间未知调整金额max','去重后时间未知循环利息max'
                       ,'去重后时间未知可用余额max','去重后时间未知预借现金额度max']
feature=pd.merge(feature, 去重后时间未知账单max,how='left', on = "用户标识")
feature['去重后时间未知上期还款金额max与时间未知上期账单金额max差值']=feature['去重后时间未知上期还款金额max']-feature['去重后时间未知上期账单金额max']
feature['去重后时间未知信用卡额度max与时间未知本期账单余额max差值']=feature['去重后时间未知信用卡额度max']-feature['去重后时间未知本期账单余额max']
feature['去重后时间未知可用余额max与时间未知预借现金额度max差值']=feature['去重后时间未知可用余额max']-feature['去重后时间未知预借现金额度max']
feature['去重后时间未知本期账单最低还款额max与时间未知循环利息max之和']=feature['去重后时间未知本期账单最低还款额max']+feature['去重后时间未知循环利息max']

去重后时间未知账单min=gb.min()
去重后时间未知账单min.columns = ['用户标识', '去重后时间未知上期账单金额min', '去重后时间未知上期还款金额min','去重后时间未知信用卡额度min'
                       ,'去重后时间未知本期账单余额min','去重后时间未知本期账单最低还款额min','去重后时间未知消费笔数min'
                       ,'去重后时间未知本期账单金额min','去重后时间未知调整金额min','去重后时间未知循环利息min'
                       ,'去重后时间未知可用余额min','去重后时间未知预借现金额度min']
feature=pd.merge(feature, 去重后时间未知账单min,how='left', on = "用户标识")
feature['去重后时间未知上期还款金额min与时间未知上期账单金额min差值']=feature['去重后时间未知上期还款金额min']-feature['去重后时间未知上期账单金额min']
feature['去重后时间未知信用卡额度min与时间未知本期账单余额min差值']=feature['去重后时间未知信用卡额度min']-feature['去重后时间未知本期账单余额min']
feature['去重后时间未知可用余额min与时间未知预借现金额度min差值']=feature['去重后时间未知可用余额min']-feature['去重后时间未知预借现金额度min']
feature['去重后时间未知本期账单最低还款额min与时间未知循环利息min之和']=feature['去重后时间未知本期账单最低还款额min']+feature['去重后时间未知循环利息min']

去重后时间未知账单mean=gb.mean()
去重后时间未知账单mean.columns = ['用户标识', '去重后时间未知上期账单金额mean', '去重后时间未知上期还款金额mean','去重后时间未知信用卡额度mean'
                        ,'去重后时间未知本期账单余额mean','去重后时间未知本期账单最低还款额mean','去重后时间未知消费笔数mean'
                        ,'去重后时间未知本期账单金额mean','去重后时间未知调整金额mean','去重后时间未知循环利息mean'
                        ,'去重后时间未知可用余额mean','去重后时间未知预借现金额度mean']
feature=pd.merge(feature, 去重后时间未知账单mean,how='left', on = "用户标识")
feature['去重后时间未知上期还款金额mean与时间未知上期账单金额mean差值']=feature['去重后时间未知上期还款金额mean']-feature['去重后时间未知上期账单金额mean']
feature['去重后时间未知信用卡额度mean与时间未知本期账单余额mean差值']=feature['去重后时间未知信用卡额度mean']-feature['去重后时间未知本期账单余额mean']
feature['去重后时间未知可用余额mean与时间未知预借现金额度mean差值']=feature['去重后时间未知可用余额mean']-feature['去重后时间未知预借现金额度mean']
feature['去重后时间未知本期账单最低还款额mean与时间未知循环利息mean之和']=feature['去重后时间未知本期账单最低还款额mean']+feature['去重后时间未知循环利息mean']


去重后时间未知账单median=gb.median()
去重后时间未知账单median.columns = ['用户标识', '去重后时间未知上期账单金额median', '去重后时间未知上期还款金额median'
                          ,'去重后时间未知信用卡额度median','去重后时间未知本期账单余额median','去重后时间未知本期账单最低还款额median'
                          ,'去重后时间未知消费笔数median','去重后时间未知本期账单金额median','去重后时间未知调整金额median'
                          ,'去重后时间未知循环利息median','去重后时间未知可用余额median','去重后时间未知预借现金额度median']
feature=pd.merge(feature, 去重后时间未知账单median,how='left', on = "用户标识")
feature['去重后时间未知上期还款金额median与时间未知上期账单金额median差值']=feature['去重后时间未知上期还款金额median']-feature['去重后时间未知上期账单金额median']
feature['去重后时间未知信用卡额度median与时间未知本期账单余额median差值']=feature['去重后时间未知信用卡额度median']-feature['去重后时间未知本期账单余额median']
feature['去重后时间未知可用余额median与时间未知预借现金额度median差值']=feature['去重后时间未知可用余额median']-feature['去重后时间未知预借现金额度median']
feature['去重后时间未知本期账单最低还款额median与时间未知循环利息median之和']=feature['去重后时间未知本期账单最低还款额median']+feature['去重后时间未知循环利息median']


去重后时间未知账单std=gb.std()
去重后时间未知账单std.columns = ['用户标识', '去重后时间未知上期账单金额std', '去重后时间未知上期还款金额std','去重后时间未知信用卡额度std'
                       ,'去重后时间未知本期账单余额std','去重后时间未知本期账单最低还款额std','去重后时间未知消费笔数std'
                       ,'去重后时间未知本期账单金额std','去重后时间未知调整金额std','去重后时间未知循环利息std'
                       ,'去重后时间未知可用余额std','去重后时间未知预借现金额度std']
feature=pd.merge(feature, 去重后时间未知账单std,how='left', on = "用户标识")
feature['去重后时间未知上期还款金额std与时间未知上期账单金额std差值']=feature['去重后时间未知上期还款金额std']-feature['去重后时间未知上期账单金额std']
feature['去重后时间未知信用卡额度std与时间未知本期账单余额std差值']=feature['去重后时间未知信用卡额度std']-feature['去重后时间未知本期账单余额std']
feature['去重后时间未知可用余额std与时间未知预借现金额度std差值']=feature['去重后时间未知可用余额std']-feature['去重后时间未知预借现金额度std']
feature['去重后时间未知本期账单最低还款额std与时间未知循环利息std之和']=feature['去重后时间未知本期账单最低还款额std']+feature['去重后时间未知循环利息std']

去重后时间未知账单var=gb.var()
去重后时间未知账单var.columns = ['用户标识', '去重后时间未知上期账单金额var', '去重后时间未知上期还款金额var','去重后时间未知信用卡额度var'
                       ,'去重后时间未知本期账单余额var','去重后时间未知本期账单最低还款额var','去重后时间未知消费笔数var'
                       ,'去重后时间未知本期账单金额var','去重后时间未知调整金额var','去重后时间未知循环利息var','去重后时间未知可用余额var','去重后时间未知预借现金额度var']
feature=pd.merge(feature, 去重后时间未知账单var,how='left', on = "用户标识")
feature['去重后时间未知上期还款金额var与时间未知上期账单金额var差值']=feature['去重后时间未知上期还款金额var']-feature['去重后时间未知上期账单金额var']
feature['去重后时间未知信用卡额度var与时间未知本期账单余额var差值']=feature['去重后时间未知信用卡额度var']-feature['去重后时间未知本期账单余额var']
feature['去重后时间未知可用余额var与时间未知预借现金额度var差值']=feature['去重后时间未知可用余额var']-feature['去重后时间未知预借现金额度var']
feature['去重后时间未知本期账单最低还款额var与时间未知循环利息var之和']=feature['去重后时间未知本期账单最低还款额var']+feature['去重后时间未知循环利息var']

feature#55596 rows × 241 columns
feature.to_csv("../feature/用户账单表特征训练表20170203_时间未知.csv",index=None,encoding="gb2312")#55596 rows × 241columns 2017.02.03

# 整体（不区分放款前放款后）：
#train=pd.read_csv("../feature/train_20170116_A.csv",encoding="gb2312")
d=训练信用卡账单表
feature=训练放款时间表[['用户标识']]
###补充特征，增加 整体的各种统计信息sum count max min mean std var等
gb=d.loc[:,['用户标识', '上期账单金额', '上期还款金额','信用卡额度','本期账单余额','本期账单最低还款额',
                                  '消费笔数','本期账单金额','调整金额','循环利息',
            '可用余额','预借现金额度','还款状态']].groupby(["用户标识"],as_index=False)

整体账单sum=gb.sum()
整体账单sum.columns = ['用户标识', '整体上期账单金额sum', '整体上期还款金额sum','整体信用卡额度sum','整体本期账单余额sum',
                     '整体本期账单最低还款额sum','整体消费笔数sum','整体本期账单金额sum','整体调整金额sum','整体循环利息sum',
                     '整体可用余额sum','整体预借现金额度sum','整体还款状态sum']
feature=pd.merge(feature, 整体账单sum,how='left', on = "用户标识")
feature['整体上期还款金额sum与整体上期账单金额sum差值']=feature['整体上期还款金额sum']-feature['整体上期账单金额sum']
feature['整体信用卡额度sum与整体本期账单余额sum差值']=feature['整体信用卡额度sum']-feature['整体本期账单余额sum']
feature['整体可用余额sum与整体预借现金额度sum差值']=feature['整体可用余额sum']-feature['整体预借现金额度sum']
feature['整体本期账单最低还款额sum与整体循环利息sum之和']=feature['整体本期账单最低还款额sum']+feature['整体循环利息sum']

整体账单count=gb.count()
整体账单count.columns = ['用户标识', '整体上期账单金额count', '整体上期还款金额count','整体信用卡额度count','整体本期账单余额count',
                     '整体本期账单最低还款额count','整体消费笔数count','整体本期账单金额count','整体调整金额count','整体循环利息count',
                     '整体可用余额count','整体预借现金额度count','整体还款状态count']
feature=pd.merge(feature, 整体账单count,how='left', on = "用户标识")

整体账单max=gb.max()
整体账单max.columns = ['用户标识', '整体上期账单金额max', '整体上期还款金额max','整体信用卡额度max','整体本期账单余额max',
                     '整体本期账单最低还款额max','整体消费笔数max','整体本期账单金额max','整体调整金额max','整体循环利息max',
                     '整体可用余额max','整体预借现金额度max','整体还款状态max']
feature=pd.merge(feature, 整体账单max,how='left', on = "用户标识")
feature['整体上期还款金额max与整体上期账单金额max差值']=feature['整体上期还款金额max']-feature['整体上期账单金额max']
feature['整体信用卡额度max与整体本期账单余额max差值']=feature['整体信用卡额度max']-feature['整体本期账单余额max']
feature['整体可用余额max与整体预借现金额度max差值']=feature['整体可用余额max']-feature['整体预借现金额度max']
feature['整体本期账单最低还款额max与整体循环利息max之和']=feature['整体本期账单最低还款额max']+feature['整体循环利息max']

整体账单min=gb.min()
整体账单min.columns = ['用户标识', '整体上期账单金额min', '整体上期还款金额min','整体信用卡额度min','整体本期账单余额min',
                     '整体本期账单最低还款额min','整体消费笔数min','整体本期账单金额min','整体调整金额min','整体循环利息min',
                     '整体可用余额min','整体预借现金额度min','整体还款状态min']
feature=pd.merge(feature, 整体账单min,how='left', on = "用户标识")
feature['整体上期还款金额min与整体上期账单金额min差值']=feature['整体上期还款金额min']-feature['整体上期账单金额min']
feature['整体信用卡额度min与整体本期账单余额min差值']=feature['整体信用卡额度min']-feature['整体本期账单余额min']
feature['整体可用余额min与整体预借现金额度min差值']=feature['整体可用余额min']-feature['整体预借现金额度min']
feature['整体本期账单最低还款额min与整体循环利息min之和']=feature['整体本期账单最低还款额min']+feature['整体循环利息min']

整体账单mean=gb.mean()
整体账单mean.columns = ['用户标识', '整体上期账单金额mean', '整体上期还款金额mean','整体信用卡额度mean','整体本期账单余额mean',
                     '整体本期账单最低还款额mean','整体消费笔数mean','整体本期账单金额mean','整体调整金额mean','整体循环利息mean',
                     '整体可用余额mean','整体预借现金额度mean','整体还款状态mean']
feature=pd.merge(feature, 整体账单mean,how='left', on = "用户标识")
feature['整体上期还款金额mean与整体上期账单金额mean差值']=feature['整体上期还款金额mean']-feature['整体上期账单金额mean']
feature['整体信用卡额度mean与整体本期账单余额mean差值']=feature['整体信用卡额度mean']-feature['整体本期账单余额mean']
feature['整体可用余额mean与整体预借现金额度mean差值']=feature['整体可用余额mean']-feature['整体预借现金额度mean']
feature['整体本期账单最低还款额mean与整体循环利息mean之和']=feature['整体本期账单最低还款额mean']+feature['整体循环利息mean']

整体账单median=gb.median()
整体账单median.columns = ['用户标识', '整体上期账单金额median', '整体上期还款金额median','整体信用卡额度median','整体本期账单余额median',
                  '整体本期账单最低还款额median','整体消费笔数median','整体本期账单金额median','整体调整金额median',
                  '整体循环利息median','整体可用余额median','整体预借现金额度median','整体还款状态median']
feature=pd.merge(feature, 整体账单median,how='left', on = "用户标识")
feature['整体上期还款金额median与整体上期账单金额median差值']=feature['整体上期还款金额median']-feature['整体上期账单金额median']
feature['整体信用卡额度median与整体本期账单余额median差值']=feature['整体信用卡额度median']-feature['整体本期账单余额median']
feature['整体可用余额median与整体预借现金额度median差值']=feature['整体可用余额median']-feature['整体预借现金额度median']
feature['整体本期账单最低还款额median与整体循环利息median之和']=feature['整体本期账单最低还款额median']+feature['整体循环利息median']

整体账单std=gb.std()
整体账单std.columns = ['用户标识', '整体上期账单金额std', '整体上期还款金额std','整体信用卡额度std','整体本期账单余额std',
                     '整体本期账单最低还款额std','整体消费笔数std','整体本期账单金额std','整体调整金额std','整体循环利息std',
                     '整体可用余额std','整体预借现金额度std','整体还款状态std']
feature=pd.merge(feature, 整体账单std,how='left', on = "用户标识")
feature['整体上期还款金额std与整体上期账单金额std差值']=feature['整体上期还款金额std']-feature['整体上期账单金额std']
feature['整体信用卡额度std与整体本期账单余额std差值']=feature['整体信用卡额度std']-feature['整体本期账单余额std']
feature['整体可用余额std与整体预借现金额度std差值']=feature['整体可用余额std']-feature['整体预借现金额度std']
feature['整体本期账单最低还款额std与整体循环利息std之和']=feature['整体本期账单最低还款额std']+feature['整体循环利息std']

整体账单var=gb.var()
整体账单var.columns = ['用户标识', '整体上期账单金额var', '整体上期还款金额var','整体信用卡额度var','整体本期账单余额var',
                     '整体本期账单最低还款额var','整体消费笔数var','整体本期账单金额var','整体调整金额var','整体循环利息var',
                     '整体可用余额var','整体预借现金额度var','整体还款状态var']
feature=pd.merge(feature, 整体账单var,how='left', on = "用户标识")
feature['整体上期还款金额var与整体上期账单金额var差值']=feature['整体上期还款金额var']-feature['整体上期账单金额var']
feature['整体信用卡额度var与整体本期账单余额var差值']=feature['整体信用卡额度var']-feature['整体本期账单余额var']
feature['整体可用余额var与整体预借现金额度var差值']=feature['整体可用余额var']-feature['整体预借现金额度var']
feature['整体本期账单最低还款额var与整体循环利息var之和']=feature['整体本期账单最低还款额var']+feature['整体循环利息var']
feature#55596 rows × 125 columns

#train=pd.read_csv("../feature/train_20170116_A.csv",encoding="gb2312")
#整体账单去重统计特征
#d=训练信用卡账单表[(训练信用卡账单表['时间']==0)]
feature_beifen=feature
#按用户标识\时间\银行标识汇总统计（去重）
data=d.loc[:,['用户标识','时间','银行标识','上期账单金额', '上期还款金额','信用卡额度','本期账单余额'
              ,'本期账单最低还款额','消费笔数','本期账单金额','调整金额','循环利息','可用余额'
              ,'预借现金额度']].groupby(["用户标识","时间","银行标识"],as_index=False).max()

gb=data.loc[:,['用户标识', '上期账单金额', '上期还款金额','信用卡额度','本期账单余额','本期账单最低还款额',
               '消费笔数','本期账单金额','调整金额','循环利息','可用余额','预借现金额度']].groupby(["用户标识"],as_index=False)

去重后整体账单sum=gb.sum()
去重后整体账单sum.columns = ['用户标识', '去重后整体上期账单金额sum', '去重后整体上期还款金额sum','去重后整体信用卡额度sum'
                       ,'去重后整体本期账单余额sum','去重后整体本期账单最低还款额sum','去重后整体消费笔数sum'
                       ,'去重后整体本期账单金额sum','去重后整体调整金额sum','去重后整体循环利息sum','去重后整体可用余额sum'
                       ,'去重后整体预借现金额度sum']
feature=pd.merge(feature, 去重后整体账单sum,how='left', on = "用户标识")
feature['去重后整体上期还款金额sum与整体上期账单金额sum差值']=feature['去重后整体上期还款金额sum']-feature['去重后整体上期账单金额sum']
feature['去重后整体信用卡额度sum与整体本期账单余额sum差值']=feature['去重后整体信用卡额度sum']-feature['去重后整体本期账单余额sum']
feature['去重后整体可用余额sum与整体预借现金额度sum差值']=feature['去重后整体可用余额sum']-feature['去重后整体预借现金额度sum']
feature['去重后整体本期账单最低还款额sum与整体循环利息sum之和']=feature['去重后整体本期账单最低还款额sum']+feature['去重后整体循环利息sum']

去重后整体账单count=gb.count()
去重后整体账单count.columns = ['用户标识', '去重后整体上期账单金额count', '去重后整体上期还款金额count','去重后整体信用卡额度count'
                         ,'去重后整体本期账单余额count','去重后整体本期账单最低还款额count','去重后整体消费笔数count'
                         ,'去重后整体本期账单金额count','去重后整体调整金额count','去重后整体循环利息count'
                         ,'去重后整体可用余额count','去重后整体预借现金额度count']
feature=pd.merge(feature, 去重后整体账单count,how='left', on = "用户标识")

去重后整体账单max=gb.max()
去重后整体账单max.columns = ['用户标识', '去重后整体上期账单金额max', '去重后整体上期还款金额max','去重后整体信用卡额度max'
                       ,'去重后整体本期账单余额max','去重后整体本期账单最低还款额max','去重后整体消费笔数max'
                       ,'去重后整体本期账单金额max','去重后整体调整金额max','去重后整体循环利息max'
                       ,'去重后整体可用余额max','去重后整体预借现金额度max']
feature=pd.merge(feature, 去重后整体账单max,how='left', on = "用户标识")
feature['去重后整体上期还款金额max与整体上期账单金额max差值']=feature['去重后整体上期还款金额max']-feature['去重后整体上期账单金额max']
feature['去重后整体信用卡额度max与整体本期账单余额max差值']=feature['去重后整体信用卡额度max']-feature['去重后整体本期账单余额max']
feature['去重后整体可用余额max与整体预借现金额度max差值']=feature['去重后整体可用余额max']-feature['去重后整体预借现金额度max']
feature['去重后整体本期账单最低还款额max与整体循环利息max之和']=feature['去重后整体本期账单最低还款额max']+feature['去重后整体循环利息max']

去重后整体账单min=gb.min()
去重后整体账单min.columns = ['用户标识', '去重后整体上期账单金额min', '去重后整体上期还款金额min','去重后整体信用卡额度min'
                       ,'去重后整体本期账单余额min','去重后整体本期账单最低还款额min','去重后整体消费笔数min'
                       ,'去重后整体本期账单金额min','去重后整体调整金额min','去重后整体循环利息min'
                       ,'去重后整体可用余额min','去重后整体预借现金额度min']
feature=pd.merge(feature, 去重后整体账单min,how='left', on = "用户标识")
feature['去重后整体上期还款金额min与整体上期账单金额min差值']=feature['去重后整体上期还款金额min']-feature['去重后整体上期账单金额min']
feature['去重后整体信用卡额度min与整体本期账单余额min差值']=feature['去重后整体信用卡额度min']-feature['去重后整体本期账单余额min']
feature['去重后整体可用余额min与整体预借现金额度min差值']=feature['去重后整体可用余额min']-feature['去重后整体预借现金额度min']
feature['去重后整体本期账单最低还款额min与整体循环利息min之和']=feature['去重后整体本期账单最低还款额min']+feature['去重后整体循环利息min']

去重后整体账单mean=gb.mean()
去重后整体账单mean.columns = ['用户标识', '去重后整体上期账单金额mean', '去重后整体上期还款金额mean','去重后整体信用卡额度mean'
                        ,'去重后整体本期账单余额mean','去重后整体本期账单最低还款额mean','去重后整体消费笔数mean'
                        ,'去重后整体本期账单金额mean','去重后整体调整金额mean','去重后整体循环利息mean'
                        ,'去重后整体可用余额mean','去重后整体预借现金额度mean']
feature=pd.merge(feature, 去重后整体账单mean,how='left', on = "用户标识")
feature['去重后整体上期还款金额mean与整体上期账单金额mean差值']=feature['去重后整体上期还款金额mean']-feature['去重后整体上期账单金额mean']
feature['去重后整体信用卡额度mean与整体本期账单余额mean差值']=feature['去重后整体信用卡额度mean']-feature['去重后整体本期账单余额mean']
feature['去重后整体可用余额mean与整体预借现金额度mean差值']=feature['去重后整体可用余额mean']-feature['去重后整体预借现金额度mean']
feature['去重后整体本期账单最低还款额mean与整体循环利息mean之和']=feature['去重后整体本期账单最低还款额mean']+feature['去重后整体循环利息mean']


去重后整体账单median=gb.median()
去重后整体账单median.columns = ['用户标识', '去重后整体上期账单金额median', '去重后整体上期还款金额median'
                          ,'去重后整体信用卡额度median','去重后整体本期账单余额median','去重后整体本期账单最低还款额median'
                          ,'去重后整体消费笔数median','去重后整体本期账单金额median','去重后整体调整金额median'
                          ,'去重后整体循环利息median','去重后整体可用余额median','去重后整体预借现金额度median']
feature=pd.merge(feature, 去重后整体账单median,how='left', on = "用户标识")
feature['去重后整体上期还款金额median与整体上期账单金额median差值']=feature['去重后整体上期还款金额median']-feature['去重后整体上期账单金额median']
feature['去重后整体信用卡额度median与整体本期账单余额median差值']=feature['去重后整体信用卡额度median']-feature['去重后整体本期账单余额median']
feature['去重后整体可用余额median与整体预借现金额度median差值']=feature['去重后整体可用余额median']-feature['去重后整体预借现金额度median']
feature['去重后整体本期账单最低还款额median与整体循环利息median之和']=feature['去重后整体本期账单最低还款额median']+feature['去重后整体循环利息median']


去重后整体账单std=gb.std()
去重后整体账单std.columns = ['用户标识', '去重后整体上期账单金额std', '去重后整体上期还款金额std','去重后整体信用卡额度std'
                       ,'去重后整体本期账单余额std','去重后整体本期账单最低还款额std','去重后整体消费笔数std'
                       ,'去重后整体本期账单金额std','去重后整体调整金额std','去重后整体循环利息std'
                       ,'去重后整体可用余额std','去重后整体预借现金额度std']
feature=pd.merge(feature, 去重后整体账单std,how='left', on = "用户标识")
feature['去重后整体上期还款金额std与整体上期账单金额std差值']=feature['去重后整体上期还款金额std']-feature['去重后整体上期账单金额std']
feature['去重后整体信用卡额度std与整体本期账单余额std差值']=feature['去重后整体信用卡额度std']-feature['去重后整体本期账单余额std']
feature['去重后整体可用余额std与整体预借现金额度std差值']=feature['去重后整体可用余额std']-feature['去重后整体预借现金额度std']
feature['去重后整体本期账单最低还款额std与整体循环利息std之和']=feature['去重后整体本期账单最低还款额std']+feature['去重后整体循环利息std']

去重后整体账单var=gb.var()
去重后整体账单var.columns = ['用户标识', '去重后整体上期账单金额var', '去重后整体上期还款金额var','去重后整体信用卡额度var'
                       ,'去重后整体本期账单余额var','去重后整体本期账单最低还款额var','去重后整体消费笔数var'
                       ,'去重后整体本期账单金额var','去重后整体调整金额var','去重后整体循环利息var','去重后整体可用余额var','去重后整体预借现金额度var']
feature=pd.merge(feature, 去重后整体账单var,how='left', on = "用户标识")
feature['去重后整体上期还款金额var与整体上期账单金额var差值']=feature['去重后整体上期还款金额var']-feature['去重后整体上期账单金额var']
feature['去重后整体信用卡额度var与整体本期账单余额var差值']=feature['去重后整体信用卡额度var']-feature['去重后整体本期账单余额var']
feature['去重后整体可用余额var与整体预借现金额度var差值']=feature['去重后整体可用余额var']-feature['去重后整体预借现金额度var']
feature['去重后整体本期账单最低还款额var与整体循环利息var之和']=feature['去重后整体本期账单最低还款额var']+feature['去重后整体循环利息var']

feature#55596 rows × 241 columns
feature.to_csv("../feature/用户账单表特征训练表20170203_整体.csv",index=None,encoding="gb2312")#55596 rows × 241columns 2017.02.03

# 5  用户账单表初级特征：
#删除指定行ok
#x=d[(d['标签']==0)].index.tolist()
#x=d.drop(x,axis=0)

#==========================================特征工程===============================================#
# d=训练信用卡账单表
# feature=训练放款时间表
#----------------------------------------放款前特征统计------------------------------------------#

#统计放款前用户上期账单金额值总额以及用户账单金额为负数的情况统计
gb=d[(d['时间']<=d['放款时间'])].groupby(["用户标识"],as_index=False)['上期账单金额']
x1=gb.apply(lambda x:x.where(x<0).count())
x2=gb.apply(lambda x:x.where(x==0.000000).count())
x=gb.agg({'放款前账单金额统计' : 'sum'})
x['放款前账单金额为负数']=x1
x['放款前账单金额为零']=x2

feature=pd.merge(feature, x,how='left', on = "用户标识")

#统计放款前用户上期还款金额值总额以及用户还款金额为负数(零)的情况统计
gb=d[(d['时间']<=d['放款时间'])].groupby(["用户标识"],as_index=False)['上期还款金额']
x1=gb.apply(lambda x:x.where(x<0).count())
x2=gb.apply(lambda x:x.where(x==0.000000).count())
x=gb.agg({'放款前还款金额统计' : 'sum'})
x['放款前还款金额为负数']=x1
x['放款前还款金额为零']=x2

feature=pd.merge(feature, x,how='left', on = "用户标识")
feature['放款前账单还款差额']=feature['放款前账单金额统计']-feature['放款前还款金额统计']

#删除0和负等异常值
d1=d[(d['上期账单金额']<=0)].index.tolist()
d=d.drop(d1,axis=0)
d2=d[(d['上期还款金额']<=0)].index.tolist()
d=d.drop(d2,axis=0)
#删除0和负等异常值后的d共1625621行

#按用户标识\时间\银行标识汇总统计
gb=d[(d['时间']<=d['放款时间'])].groupby(["用户标识","时间","银行标识"],as_index=False)
x1=gb['上期账单金额'].agg({'放款前该用户该银行上月账单金额总计' : 'sum','放款前该用户该银行上月账单金额最大值' : 'max'})
x2=gb['上期还款金额'].agg({'放款前该用户该银行上月还款金额总计' : 'sum','放款前该用户该银行还款金额最大值' : 'max'})
x3=gb['消费笔数'].agg({'用户放款前消费笔数最大值' : 'max'})
x4=gb['循环利息'].agg({'用户放款前循环利息最大值' : 'max'})

gb1=x1.groupby(["用户标识"],as_index=False)
gb2=x2.groupby(["用户标识"],as_index=False)
gb3=x3.groupby(["用户标识"],as_index=False)
gb4=x4.groupby(["用户标识"],as_index=False)

x11=gb1['放款前该用户该银行上月账单金额总计'].agg({'放款前该用户账单金额汇总(去重)' : 'sum','放款前该用户账单数(去重)' : 'count'})
x12=gb1['放款前该用户该银行上月账单金额最大值'].agg({'放款前该用户账单金额最大值汇总(去重)' : 'sum'})

x21=gb2['放款前该用户该银行上月还款金额总计'].agg({'放款前该用户账单还款金额汇总(去重)' : 'sum'})
x22=gb2['放款前该用户该银行还款金额最大值'].agg({'放款前该用户账单还款金额最大值汇总(去重)' : 'sum'})

x31=gb3['用户放款前消费笔数最大值'].agg({'用户放款前消费笔数(去重)' : 'sum'})
x41=gb4['用户放款前循环利息最大值'].agg({'用户放款前循环利息(去重)' : 'sum'})

feature=pd.merge(feature, x11,how='left', on = "用户标识")
feature=pd.merge(feature, x12,how='left', on = "用户标识")
feature=pd.merge(feature, x21,how='left', on = "用户标识")
feature=pd.merge(feature, x22,how='left', on = "用户标识")
feature=pd.merge(feature, x31,how='left', on = "用户标识")
feature=pd.merge(feature, x41,how='left', on = "用户标识")

x=pd.merge(x1, x2,how='inner')
gb3=x[(x['放款前该用户该银行上月账单金额最大值']>x['放款前该用户该银行还款金额最大值'])].groupby(["用户标识"],as_index=False)
gb4=x[(x['放款前该用户该银行上月账单金额最大值']==x['放款前该用户该银行还款金额最大值'])].groupby(["用户标识"],as_index=False)
gb5=x[(x['放款前该用户该银行上月账单金额最大值']<x['放款前该用户该银行还款金额最大值'])].groupby(["用户标识"],as_index=False)

x31=gb3['用户标识'].agg({'放款前账单大于还款计数(去重)' : 'count'})
x32=gb4['用户标识'].agg({'放款前账单等于还款计数(去重)' : 'count'})
x33=gb5['用户标识'].agg({'放款前账单小于还款计数(去重)' : 'count'})

feature=pd.merge(feature, x31,how='left', on = "用户标识")
feature=pd.merge(feature, x32,how='left', on = "用户标识")
feature=pd.merge(feature, x33,how='left', on = "用户标识")

feature['放款前账单汇总还款差额(去重)']=feature['放款前该用户账单金额汇总(去重)']-feature['放款前该用户账单还款金额汇总(去重)']
feature['放款前账单最大值还款差额(去重)']=feature['放款前该用户账单金额最大值汇总(去重)']-feature['放款前该用户账单还款金额最大值汇总(去重)']

#统计放款前用户消费笔数，循环利息总计
gb=d[(d['时间']<=d['放款时间'])].groupby(["用户标识"],as_index=False)
x1=gb['消费笔数'].agg({'用户放款前消费笔数' : 'sum'})
x2=gb['循环利息'].agg({'用户放款前循环利息' : 'sum'})
x3=gb['信用卡额度'].agg({'用户放款前信用卡额度最大值' : 'max'})

feature=pd.merge(feature, x1,how='left', on = "用户标识")
feature=pd.merge(feature, x2,how='left', on = "用户标识")
feature=pd.merge(feature, x3,how='left', on = "用户标识")

#----------------------------------------放款后特征统计------------------------------------------#