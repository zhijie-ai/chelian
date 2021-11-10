#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/8/5 15:25                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

# 注意，LabelBinarizer对二值的onehot编码不太友好，只有一个值了
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

df = pd.DataFrame({'userid':[1,5,6,7,14,20,22,23,34,36],
                   'gender':[1,1,1,1,2,1,2,1,1,1],
                   'state':[1,1,2,2,1,2,1,1,2,0]})


def multi_column_encoder(df,column_name_list):
    ohe = OneHotEncoder()
    lb = LabelEncoder()
    for column_name in column_name_list:
        lb.fit_transform(df[column_name])
        print(lb.classes_)
        df_dummies = pd.DataFrame(ohe.fit_transform(df[[column_name]]).todense(), columns = list(lb.classes_))
        df_dummies.rename(columns=lambda x: column_name + "_" + str(x), inplace=True)
        df = pd.concat([df, df_dummies], axis=1)
        df.drop(column_name, axis=1, inplace=True)
    return df

def multi_column_encoder2(df,column_name_list):
    lb = LabelBinarizer()
    le = LabelEncoder()
    for column_name in column_name_list:
        le.fit_transform(df[column_name])
        df_dummies = pd.DataFrame(lb.fit_transform(df[column_name]), columns = list(le.classes_))
        df_dummies.rename(columns=lambda x: column_name + "_" + str(x), inplace=True)
        df = pd.concat([df, df_dummies], axis=1)
        df.drop(column_name, axis=1, inplace=True)
    return df

df = multi_column_encoder2(df,['gender','state'])
print(df)
