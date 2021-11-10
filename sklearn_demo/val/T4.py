#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/8/5 17:39                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import pandas as pd
import numpy as np

df = pd.DataFrame({'userid':[50,5,6,7,14,20,22,23,34,36],
                   'state':list(map(str,[1,1,2,2,1,2,1,1,2,4])),
                   'gender':list(map(str,[1,1,1,1,2,1,2,1,1,3])),
                   'age':[20,30,10,50,60,10,20,80,90,40],
                   'sex':['boy','girl','girl','boy','girl','boy','girl','girl','boy',np.nan],
                   'height':[2340, 2340, 736, 2400, 812, 2340, 2340, 2160, 736, 2160],
                   'width':[1080, 1080, 414, 1080, 375, 1080, 1080, 1080, 414, 1080],
                   'login_cnt_30':[0,10,20,0,11,55,0,6,30,50],
                   })

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,Normalizer
from sklearn.base import BaseEstimator, TransformerMixin


class DeLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,cols=[]):
        self.cols = cols

    def fit(self,X=None,y=None):
        return self


    def transform(self,X,y=None):
        print(type(X),self.cols)
        for col in self.cols:
            lb = LabelEncoder()
            X[col] =lb.fit_transform(X[col])
        return X

class HandleResolution(BaseEstimator, TransformerMixin):
    def __init__(self,cols):
        self.cols = cols

    def fit(self,X=None,y=None):
        return self


    def transform(self,X,y=None):
        X['new_resolution'] = X[self.cols[0]].map(str)+'*'+X[self.cols[1]].map(str)
        X.drop([self.cols[0],self.cols[1]],axis=1,inplace=True)
        return X






label_cols =['gender','state']
cat_cols =['gender','state']
num_cols = ['login_cnt_30']
new_cols = ['width','height']
numeric_transformer = Pipeline(steps=[
    # ('scaler', StandardScaler()),
    # ('minmax',MinMaxScaler()),
    ('aaa',Normalizer())])


categorical_transformer = Pipeline(steps=[
    # ('imputer', SimpleImputer(strategy='constant', fill_value=100)),
    ('onehot', OneHotEncoder(handle_unknown='ignore',sparse=False))])

lb_transformer = Pipeline(steps=[
    ('lb_imputer',SimpleImputer(strategy='constant',fill_value='unknown')),
    ('lb', DeLabelEncoder(label_cols))])# 如果构造函数中不传值，没效果

new_feat_transformer = Pipeline(steps=[
    ('lb_new', HandleResolution(new_cols))])# 如果构造函数中不传值，没效果



preprocessor = ColumnTransformer(
    transformers = [
        # ('new',new_feat_transformer,new_cols),# 如果不传值，报错
        # ('lb',lb_transformer,label_cols),# 如果不传值，报错

        # ('num',numeric_transformer,num_cols),
        ('cat',categorical_transformer,cat_cols)
    ],
        # remainder = 'passthrough'

)

arr = preprocessor.fit_transform(df)
print(arr)

'''
{'userid':[50,5,6,7,14,20,22,23,34,36],
                   'state':list(map(str,[1,1,2,2,1,2,1,1,2,4])),
                   'gender':list(map(str,[1,1,1,1,2,1,2,1,1,3])),
                   'age':[20,30,10,50,60,10,20,80,90,40],
                   'sex':['boy','girl','girl','boy','girl','boy','girl','girl','boy',np.nan],
                   'height':[2340, 2340, 736, 2400, 812, 2340, 2340, 2160, 736, 2160],
                   'width':[1080, 1080, 414, 1080, 375, 1080, 1080, 1080, 414, 1080],
                   'login_cnt_30':[0,10,20,0,11,55,0,6,30,50],
                   }
'''
df = pd.DataFrame({'userid':[50],
                   'state':list(map(str,[5])),
                   'gender':list(map(str,[6])),
                   'age':[20],
                   'sex':['boy'],
                   'height':[2340],
                   'width':[1080],
                   'login_cnt_30':[0],
                   })
print(df)
print(preprocessor.transform(df))
print(preprocessor)