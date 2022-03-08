#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/8/11 11:01                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

np.random.seed(10)
df = pd.DataFrame({'age':np.random.randint(0,40,10),
                   'fare':np.random.randn(10),
                   'embarked':list(map(str,np.random.randint(0,3,10))),
                   'sex':list(map(str,np.random.randint(0,2,10))),
                   'pclass':list(map(str,np.random.randint(0,3,10)))})

numeric_features = ['age', 'fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['embarked', 'sex', 'pclass']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ], remainder='passthrough')

np.set_printoptions(suppress=True,   precision=10,  threshold=2000,  linewidth=5000)
ds = preprocessor.fit_transform(df)
print(ds)
print(preprocessor._columns)
# print(preprocessor.named_transformers_)
print(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features))
# print(preprocessor.named_transformers_['num'][-1].get_feature_names_out(numeric_features))
# print(preprocessor.named_transformers_['num'][-1])
# print(preprocessor.named_transformers_['num']['scaler'].get_feature_names_out(numeric_features))
# print(preprocessor.named_transformers_['cat']['onehot'])