#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author    : zushoujie
# file      : T.py
# time      : 2022/1/29 15:53
# file_desc :
# Copyright  2022 云积分科技. All rights reserved.

# https://blog.csdn.net/u012735708/article/details/83749703
# 采用的方法是参数之间是没有依赖关系的

import time

import pandas as pd
import xgboost as xgb

def train_model(X_train, y_train, X_val, y_val):
    print('start model training.......{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
    dtrain = xgb.DMatrix(X_train, y_train)
    # 交叉验证(调参)
    print('交叉验证')
    best_params = {}

    print('设置参数....')
    # 第一步:xgb.cv确定最优的迭代次数
    t1 = time.time()
    params = {
        'booster': 'gbtree',
        'gamma': 0.05,  # 树的叶子节点下一个区分的最小损失，越大算法模型越保守
        'max_depth': 12,
        'lambda': 450,  # l2正则项权重
        'subsample': 0.4,  # 采样训练数据，设置为0.5
        'colsample_bytree': 0.7,  # 构建树时采样比例
        'min_child_weight': 12,  # 节点的最少特征树
        # 'verbosity':0,#为1时模型运行不输出。
        'eta': 0.005,  # 类似学习率
        'seed': 2,
        # 'tree_method': 'gpu_hist',
        'eval_metric': ['auc'],
    }
    cv_results = xgb.cv(params, dtrain, num_boost_round=1000, early_stopping_rounds=50, nfold=3)
    print('best_estimators:', len(cv_results['test-auc-mean']))
    best_params['n_estimators'] = len(cv_results['test-auc-mean'])
    t2 = time.time()
    print('交叉验证花费的时间为:{} m,best_params:{}'.format((t2 - t1) / 60, best_params))

    # 准确率
    print('调参1,提高准确率max_depth:{}'.format(list(range(3, 8, 1))))
    max_auc = -1
    params_ = params.copy()
    for idx, max_depth in enumerate(range(3, 8, 1)):
        t_ = time.time()
        params_['max_depth'] = max_depth

        cv_results = xgb.cv(params_,
                            dtrain,
                            seed=1,
                            nfold=3,
                            early_stopping_rounds=10)

        t_2 = time.time()
        mean_auc = pd.Series(cv_results['test-auc-mean']).max()
        print('score: %.6f,%d,time cost:%d' % (mean_auc, idx, (t_2 - t_)))

        if mean_auc >= max_auc:
            max_auc = mean_auc
            best_params['max_depth'] = max_depth

    t3 = time.time()
    print('调参1,num_leaves,max_depth 花费的时间 :{} m,best_params:{}'.format((t3 - t2) / 60, best_params))

    print('调参2,提高准确率,learning_rate:{}'.format([0.01, 0.02, 0.05, 0.1, 0.15]))
    max_auc = -1
    params_ = params.copy()
    for learning_rate in [0.01, 0.02, 0.05, 0.1, 0.15]:
        params_['learning_rate'] = learning_rate

        cv_results = xgb.cv(params_,
                            dtrain,
                            seed=1,
                            nfold=3,
                            early_stopping_rounds=10)
        mean_auc = pd.Series(cv_results['test-auc-mean']).max()

        if mean_auc >= max_auc:
            max_auc = mean_auc
            best_params['learning_rate'] = learning_rate

    t4 = time.time()
    print('调参2,learning_rate花费的时间 :{} m,best_params:{}'.format((t4 - t3) / 60, best_params))

    # 过拟合
    print("调参3：降低过拟合,colsample_bytree:{},\nsubsample:{}".format([0.6, 0.8, 1.0], [0.6, 0.8, 1.0]))
    max_auc = -1
    params_ = params.copy()
    for colsample_bytree in [0.6, 0.8, 1.0]:
        for subsample in [0.6, 0.8, 1.0]:
            params_['colsample_bytree'] = colsample_bytree
            params_['subsample'] = subsample

            cv_results = xgb.cv(
                params_,
                dtrain,
                seed=1,
                nfold=3,
                early_stopping_rounds=10)

            mean_auc = pd.Series(cv_results['test-auc-mean']).max()

            if mean_auc >= max_auc:
                max_auc = mean_auc
                best_params['colsample_bytree'] = colsample_bytree
                best_params['subsample'] = subsample

    t5 = time.time()
    print('调参3：colsample_bytree,subsample 花费的时间 :{} m,best_params:{}'.format((t5 - t4) / 60, best_params))

    print("调参4：降低过拟合:reg_alpha:{},\t reg_lambda:{},\n gamma:{}".format([1e-5, 0.0, 0.1, 0.5, 1.0],
                                                                       [1e-5, 0.0, 0.1, 0.5, 1.0], [1, 6, 11, 16, 21]))
    max_auc = -1
    params_ = params.copy()
    for reg_alpha in [1e-5, 0.0, 0.1, 0.5, 1.0]:
        for reg_lambda in [1e-5, 0.0, 0.1, 0.5, 1.0]:
            for gamma in [1, 6, 11, 16]:
                params_['reg_alpha'] = reg_alpha
                params_['reg_lambda'] = reg_lambda
                params_['gamma'] = gamma

                cv_results = xgb.cv(
                    params_,
                    dtrain,
                    seed=1,
                    nfold=3,
                    early_stopping_rounds=10)

                mean_auc = pd.Series(cv_results['test-auc-mean']).max()

                if mean_auc >= max_auc:
                    max_auc = mean_auc
                    best_params['reg_alpha'] = reg_alpha
                    best_params['reg_lambda'] = reg_lambda
                    best_params['gamma'] = gamma

    t6 = time.time()
    print('调参4：lambda_l1,lambda_l2 gamma花费的时间 :{} m,best_params:{}'.format((t6 - t5) / 60, best_params))

    print(
        "调参5：降低过拟合2,建树过程中的增益相关,min_child_weight:{},\t max_delta_step:{}".format([0, 2, 5, 10, 20], [0, 0.2, 0.6, 1, 2]))
    max_auc = -1
    params_ = params.copy()
    for min_child_weight in [0, 2, 5, 10, 20]:
        for max_delta_step in [0, 0.2, 0.6, 1, 2]:
            params_['min_child_weight'] = min_child_weight
            params_['max_delta_step'] = max_delta_step

            cv_results = xgb.cv(
                params_,
                dtrain,
                seed=1,
                nfold=3,
                early_stopping_rounds=10)

            mean_auc = pd.Series(cv_results['test-auc-mean']).max()

            if mean_auc >= max_auc:
                max_auc = mean_auc
                best_params['min_child_weight'] = min_child_weight
                best_params['max_delta_step'] = max_delta_step

    t7 = time.time()
    print('调参5：min_child_weight,max_delta_step 降低过拟合花费的时间 :{} m,best_params:{}'.format((t7 - t6) / 60, best_params))

    print("调参6：设置样本不均衡比例 scale_pos_weight:{}".format([1, 10, 20, 30, 40, 50]))
    max_auc = -1
    params_ = params.copy()
    for scale_pos_weight in [1, 10, 20, 30, 40, 50]:
        params_['scale_pos_weight'] = scale_pos_weight

        cv_results = xgb.cv(
            params_,
            dtrain,
            seed=1,
            nfold=3,
            early_stopping_rounds=10)

        mean_auc = pd.Series(cv_results['test-auc-mean']).max()

        if mean_auc >= max_auc:
            max_auc = mean_auc
            best_params['scale_pos_weight'] = scale_pos_weight

    t8 = time.time()
    print('调参6：scale_pos_weight降低过拟合花费的时间 :{} m,best_params:{}'.format((t8 - t7) / 60, best_params))

    print('交叉验证花费的总时间 :{} m'.format((t8 - t1) / 60))
    print('best_params:{}'.format(best_params))

    # 使用最优参数训练模型
    print('使用最优参数训练模型.......')

    model = xgb.XGBClassifier(**best_params, use_label_encoder=False)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', early_stopping_rounds=20, verbose=False)
    t9 = time.time()
    print('使用最优参数训练模型花费的时间:{} m'.format((t9 - t8) / 60))
    return model, best_params


if __name__ == '__main__':
    import numpy as np
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    import datetime
    import pickle

    t1 = time.time()
    train = pd.read_csv('handled/train.csv')
    test = pd.read_csv('handled/test.csv')
    val = pd.read_csv('handled/val.csv')
    X_train = train.drop('label', axis=1)
    X_test = test.drop('label', axis=1)
    X_val = val.drop('label', axis=1)
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
    with open(model_path + 'xgb_model_param.pickle', 'wb') as f:
        pickle.dump(model, f)

    y_pred = model.predict(X_test, iteration_range=model.best_ntree_limit)
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
    
