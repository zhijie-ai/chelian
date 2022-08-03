#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author    : zushoujie
# file      : xgb_predict.py
# time      : 2021/11/26 17:09
# file_desc :
# Copyright  2021 All rights reserved.
import xgboost as xgb

from general_config import ModelConfig
import pickle
import pandas as pd
import numpy as np

xgb_moel_path = ModelConfig.XGB_MODEL_PATH + 'xgb_model_param.pickle'
model = pickle.load(open(xgb_moel_path, 'rb'))

X_val_ = pd.read_csv(ModelConfig.XGB_DATA_PATH + 'X_val.csv')
y_val_ = pd.read_csv(ModelConfig.XGB_DATA_PATH + 'y_val.csv', header=None)

d_val = xgb.DMatrix(X_val_[:10], label=y_val_[:10])
y_pred = model.predict(X_val_[:10], ntree_limit=model.best_ntree_limit)
print(y_pred)
print(model.get_booster().predict(d_val))
# model.get_booster().get_num_boosting_rounds()
print('=========================pred_leaf=True===============')
pred = model.get_booster().predict(d_val, pred_leaf=True)
print(np.array(pred).shape, np.max(pred))
print('=========================pred_contribs=True===============')
pred = model.get_booster().predict(d_val, pred_contribs=True)
print(pred)
print(len(pred), len(pred[0]))
s = sum(pred[0])
print(s)
print(1/float(1+np.exp(-s)))
