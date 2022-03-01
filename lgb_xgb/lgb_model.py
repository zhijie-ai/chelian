#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author    : zushoujie
# file      : T.py
# time      : 2022/1/29 15:53
# file_desc :
# Copyright  2022 云积分科技. All rights reserved.


import time

import pandas as pd
import lightgbm as lgb
import pandas as pd
from sklearn import metrics

def train_model(X_train, y_train, X_val, y_val):
    ### 数据转换
    print('数据转换')
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)

    ### 设置初始参数--不含交叉验证参数
    print('设置参数')
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'nthread':4,
        'learning_rate':0.1,
        'verbose':-1
    }

    ### 交叉验证(调参)
    print('交叉验证')
    max_auc = float('0')
    best_params = {}

    # 准确率
    print("调参1：提高准确率")
    for num_leaves in range(5,100,5):
        for max_depth in range(3,8,1):
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth

            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                nfold=5,
                metrics=['auc'],
                early_stopping_rounds=10,
                verbose_eval=False
            )

            mean_auc = pd.Series(cv_results['auc-mean']).max()
            boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()

            if mean_auc >= max_auc:
                max_auc = mean_auc
                best_params['num_leaves'] = num_leaves
                best_params['max_depth'] = max_depth
    if 'num_leaves' and 'max_depth' in best_params.keys():
        params['num_leaves'] = best_params['num_leaves']
        params['max_depth'] = best_params['max_depth']

    # 过拟合
    print("调参2：降低过拟合")
    for max_bin in range(5,256,10):
        for min_data_in_leaf in range(1,102,10):
            params['max_bin'] = max_bin
            params['min_data_in_leaf'] = min_data_in_leaf
            lgb_train = lgb.Dataset(X_train, y_train)

            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                nfold=5,
                metrics=['auc'],
                early_stopping_rounds=10,
                verbose_eval=False
            )

            mean_auc = pd.Series(cv_results['auc-mean']).max()
            boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()

            if mean_auc >= max_auc:
                max_auc = mean_auc
                best_params['max_bin']= max_bin
                best_params['min_data_in_leaf'] = min_data_in_leaf
    if 'max_bin' and 'min_data_in_leaf' in best_params.keys():
        params['min_data_in_leaf'] = best_params['min_data_in_leaf']
        params['max_bin'] = best_params['max_bin']

    print("调参3：降低过拟合")
    for feature_fraction in [0.6,0.7,0.8,0.9,1.0]:
        for bagging_fraction in [0.6,0.7,0.8,0.9,1.0]:
            for bagging_freq in range(0,50,5):
                params['feature_fraction'] = feature_fraction
                params['bagging_fraction'] = bagging_fraction
                params['bagging_freq'] = bagging_freq
                lgb_train = lgb.Dataset(X_train, y_train)

                cv_results = lgb.cv(
                    params,
                    lgb_train,
                    seed=1,
                    nfold=5,
                    metrics=['auc'],
                    early_stopping_rounds=10,
                    verbose_eval=False
                )

                mean_auc = pd.Series(cv_results['auc-mean']).max()
                boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()

                if mean_auc >= max_auc:
                    max_auc=mean_auc
                    best_params['feature_fraction'] = feature_fraction
                    best_params['bagging_fraction'] = bagging_fraction
                    best_params['bagging_freq'] = bagging_freq

    if 'feature_fraction' and 'bagging_fraction' and 'bagging_freq' in best_params.keys():
        params['feature_fraction'] = best_params['feature_fraction']
        params['bagging_fraction'] = best_params['bagging_fraction']
        params['bagging_freq'] = best_params['bagging_freq']


    print("调参4：降低过拟合")
    for lambda_l1 in [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]:
        for lambda_l2 in [1e-5,1e-3,1e-1,0.0,0.1,0.4,0.6,0.7,0.9,1.0]:
            params['lambda_l1'] = lambda_l1
            params['lambda_l2'] = lambda_l2
            lgb_train = lgb.Dataset(X_train, y_train)

            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                nfold=5,
                metrics=['auc'],
                early_stopping_rounds=10,
                verbose_eval=False
            )

            mean_auc = pd.Series(cv_results['auc-mean']).max()
            boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()

            if mean_auc >= max_auc:
                max_auc=mean_auc
                best_params['lambda_l1'] = lambda_l1
                best_params['lambda_l2'] = lambda_l2
    if 'lambda_l1' and 'lambda_l2' in best_params.keys():
        params['lambda_l1'] = best_params['lambda_l1']
        params['lambda_l2'] = best_params['lambda_l2']

    print("调参5：降低过拟合2")
    for min_split_gain in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        params['min_split_gain'] = min_split_gain
        lgb_train = lgb.Dataset(X_train, y_train)

        cv_results = lgb.cv(
            params,
            lgb_train,
            seed=1,
            nfold=5,
            metrics=['auc'],
            early_stopping_rounds=10,
            verbose_eval=False
        )

        mean_auc = pd.Series(cv_results['auc-mean']).max()
        boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()

        if mean_auc >= max_auc:
            max_auc=mean_auc

            best_params['min_split_gain'] = min_split_gain
    if 'min_split_gain' in best_params.keys():
        params['min_split_gain'] = best_params['min_split_gain']

    print('best params',params)
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', early_stopping_rounds=20, verbose=False)
    return model, best_params


if __name__ == '__main__':
    import numpy as np
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    import datetime
    import pickle

    t1 = time.time()
    train = pd.read_csv('handled/train_range.csv')
    test = pd.read_csv('handled/test_range.csv')
    val = pd.read_csv('handled/val_range.csv')
    X_train = train.drop(['user_id','label'], axis=1)
    X_test = test.drop(['user_id','label'], axis=1)
    X_val = val.drop(['user_id','label'], axis=1)
    y_train = train.label
    y_test = test.label
    y_val = val.label
    t2 = time.time()
    print('load data cost time:{} m'.format((t2 - t1) / 60))
    print('X_train.shape,X_test.shape,X_val.shape,y_train.shape,y_test.shape,y_val.shape')
    print(X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape)

    model, best_param = train_model(X_train, y_train, X_val, y_val)
    print('best_param', best_param)
    model_path = 'model/'
    with open(model_path + 'lgb_model_param.pickle', 'wb') as f:
        pickle.dump(model, f)

    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    score = f1_score(y_test, y_pred)
    print('auc:{}'.format(auc))
    print('acc:{}'.format(acc))
    print('precision:{}'.format(precision))
    print('recall:{}'.format(recall))
    print('f1_score:{}'.format(score))
