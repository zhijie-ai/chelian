#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/8/6 16:32                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris

df = pd.DataFrame({'userid':[50,5,6,7,14,20,22,23,34,36],
                   'state':[1,1,2,2,1,2,1,1,2,4],
                   'gender':[1,1,1,1,2,1,2,1,1,3],
                   'age':[20,30,10,50,60,10,20,80,90,40],
                   'sex':['boy','girl','girl','boy','girl','boy','girl','girl','boy',np.nan],
                   'height':[2340, 2340, 736, 2400, 812, 2340, 2340, 2160, 736, 2160],
                   'width':[1080, 1080, 414, 1080, 375, 1080, 1080, 1080, 414, 1080],
                   'login_cnt_30':[0,10,20,0,11,55,0,6,30,30],
                   'app_version':['3.21.2','3.21.3','3.21.4','3.21.5','3.21.6','3.21.7',
                                  '3.21.8','3.21.9','3.21.1','3.21.10']
                   })

# df = pd.read_csv('dataset_all_sampled_10000.csv',index_col=None,nrows=10)

class TopFeaturesSelector(BaseEstimator,TransformerMixin):
    def __init__(self,feature_importance_k=5):
        self.top_k_attributes=None
        self.k=feature_importance_k
    def fit(self,X,y):
        reg=RandomForestRegressor()
        reg.fit(X,y)
        feature_importance=reg.feature_importances_
        top_k_attributes=np.argsort(-feature_importance)[0:self.k]
        self.top_k_attributes=top_k_attributes
        return(self)
    def transform(self, X,**fit_params):
        return(X[:,self.top_k_attributes])

'''
prepare_and_top_feature_pipeline=Pipeline([
    ('feature_selector',TopFeaturesSelector(feature_importance_k=5))
])
iris = load_iris()
res = prepare_and_top_feature_pipeline.fit_transform(iris.data,iris.target)
'''


ct = ColumnTransformer(
    transformers=[
        ('imp',SimpleImputer(strategy='median'),['state','login_cnt_30']),
        ('lb',LabelEncoder(),['state'])
    ])

ct.fit(df)
res = ct.transform(df)
print(res)