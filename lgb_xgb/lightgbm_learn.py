#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/8/16 11:01
 =================知行合一=============
'''
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris

# dataset = pd.read_csv('./data/dataset.csv',encoding='gbk')
# y=dataset['故障码Indexer']
# X = dataset.drop(['故障码Indexer'],axis=1)
#
# # 将字符串类型转换为数值类型及标准归一化操作
# X = pd.get_dummies(X)
# standardScaler = StandardScaler()
# X = standardScaler.fit_transform(X)


# X_gen, y_gen = make_classification(50000, n_classes=2, n_clusters_per_class=2, n_informative=4)
# X_train,X_test,y_train,y_test = train_test_split(X_gen,y_gen,test_size=0.2)
#
#
iris = load_iris()
X, y = iris.data,iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

def lightgbm_sklearn():
    from lightgbm.sklearn import LGBMClassifier
    import lightgbm as lgb

    start = time.time()
    # lgbm = LGBMClassifier(n_estimators=100)
    lgbm  = lgb.LGBMClassifier(n_estimators=100)
    lgbm.fit(X_train, y_train)
    end = time.time()
    score = lgbm.score(X_test, y_test)

    print('lightgbm 算法训练花费的时间为:', (end - start))#500左右
    print('score:',score)#0.00338711439706
    y_pred = lgbm.predict(X_test)
    print('y_pred',y_pred)

def lightgbm():
    import lightgbm as lgb
    import numpy as np

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'multiclass',  # 目标函数
        'metric': {'l2', 'acc'},  # 评估函数
        'num_class':3,
        # 'predict_raw_score':True,
        # 'num_leaves': 31,  # 叶子节点数
        'learning_rate': 0.05,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        # 'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }

    train_data = lgb.Dataset(X_train,label=y_train)
    test_data = lgb.Dataset(X_test,label=y_test)

    start = time.time()
    lgb_model = lgb.train(params,train_set = train_data)
    end = time.time()
    y_pred = lgb_model.predict(X_test)
    pred_label = np.argmax(y_pred,axis=1)
    print('lightgbm 算法训练花费的时间为:', (end - start))# 0.7750442028045654
    print('y_pred:', y_pred)
    print('pred_label:',pred_label)
    print('y:',y_test)


#官网的例子
def example():
    # coding: utf-8
    # pylint: disable = invalid-name, C0111
    import lightgbm as lgb
    import pandas as pd
    from sklearn.metrics import mean_squared_error

    # load or create your dataset
    print('Load data...')
    df_train = pd.read_csv('./data/regression.train', header=None, sep='\t')
    df_test = pd.read_csv('./data/regression.RL', header=None, sep='\t')

    y_train = df_train[0].values
    y_test = df_test[0].values
    X_train = df_train.drop(0, axis=1).values
    X_test = df_test.drop(0, axis=1).values

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'auc'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=5)

    print('Save model...')
    # save model to file
    gbm.save_model('model.txt')

    print('Start predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # eval
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    print('y_pred:',y_pred)


def example2():
    # coding: utf-8
    # pylint: disable = invalid-name, C0111
    import numpy as np
    import pandas as pd
    import lightgbm as lgb

    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GridSearchCV

    # load or create your dataset
    print('Load data...')
    df_train = pd.read_csv('./data/regression.train', header=None, sep='\t')
    df_test = pd.read_csv('./data/regression.RL', header=None, sep='\t')

    y_train = df_train[0].values
    y_test = df_test[0].values
    X_train = df_train.drop(0, axis=1).values
    X_test = df_test.drop(0, axis=1).values

    print('Start training...')
    # train
    gbm = lgb.LGBMRegressor(objective='regression',
                            num_leaves=31,
                            learning_rate=0.05,
                            n_estimators=20)
    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='l1',
            early_stopping_rounds=5)

    print('Start predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    print('*'*8)
    print(y_pred)
    # eval
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

    # feature importances
    print('Feature importances:', list(gbm.feature_importances_))

    # self-defined eval metric
    # f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
    # Root Mean Squared Logarithmic Error (RMSLE)
    def rmsle(y_true, y_pred):
        return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False

    print('Start training with custom eval function...')
    # train
    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=rmsle,
            early_stopping_rounds=5)

    print('Start predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    print('y_pred:,',y_pred)
    # eval
    print('The rmsle of prediction is:', rmsle(y_test, y_pred)[1])

    # other scikit-learn modules
    estimator = lgb.LGBMRegressor(num_leaves=31)

    param_grid = {
        'learning_rate': [0.01, 0.1, 1],
        'n_estimators': [20, 40]
    }

    gbm = GridSearchCV(estimator, param_grid)

    gbm.fit(X_train, y_train)

    print('Best parameters found by grid search are:', gbm.best_params_)


if __name__ == '__main__':
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier

    lightgbm()
    # lightgbm_sklearn()
    # example()
    # example2()