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
from sklearn.datasets import dump_svmlight_file

df = pd.DataFrame({'userid':[50,5,6,7,14,20,22,23,34,36],
                   'state':[1,1,2,2,1,2,1,1,2,4],
                   'gender':[1,1,1,1,2,1,2,1,1,3],
                   'age':[20,30,10,50,60,10,20,80,90,40],
                   'sex':['boy','girl','girl','boy','girl','boy','girl','girl','boy','girl'],
                   'height':[2340, 2340, 736, 2400, 812, 2340, 2340, 2160, 736, 2160],
                   'width':[1080, 1080, 414, 1080, 375, 1080, 1080, 1080, 414, 1080],
                   'login_cnt_30':[0,10,20,0,11,55,0,6,30,40]
                   })

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,Normalizer
import numpy as np

cat_cols =['gender','state','sex']
num_cols = ['login_cnt_30']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=100)),
    # ('scaler', StandardScaler())
])


"""
handle_unknown : {'error', 'ignore'}, default='error'
        Whether to raise an error or ignore if an unknown categorical feature
        is present during transform (default is to raise). When this parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros. In the inverse transform, an unknown category
        will be denoted as None.
"""
categorical_transformer = Pipeline(steps=[
    # ('imputer', SimpleImputer(strategy='constant', fill_value=100)),
    ('onehot', OneHotEncoder(handle_unknown='ignore',sparse=False, dtype=np.int))])


preprocessor = ColumnTransformer(
    transformers = [
        # ('num',numeric_transformer,num_cols),
        ('cat',categorical_transformer,cat_cols)],
    remainder = 'drop'

)

np.set_printoptions(suppress=True,   precision=10,  threshold=2000,  linewidth=5000)
arr = preprocessor.fit_transform(df,range(100,110))

dump_svmlight_file(arr,range(100,110),'test.libsvm')
print(np.around(arr,decimals=1))
preprocessor.fit(df)
data = df.iloc[1:10,:]
# print(data)
# print(preprocessor.transform(data))
# print(preprocessor)