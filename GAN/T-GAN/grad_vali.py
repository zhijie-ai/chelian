#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Tencent Inc.
# Author: ranchodzu (ranchodzu@tencent.com)
# TIME: 2023/1/16 11:13
# predict和model()方法梯度验证

# https://blog.csdn.net/weixin_39653948/article/details/105816295
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow.keras.backend as K


def build_discriminator():
    model = Sequential()

    model.add(Dense(10, input_shape=(10,)))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=(10,))
    validity = model(img)
    model = Model(img, validity)
    return model

def train(DIS, batch_size=10):
    w = tf.Variable([2., 5.])
    u = tf.Variable([7., 9.])

    x = 2 * w + 3 * u

    # Load the dataset
    X_train = np.random.randn(batch_size, 10)
    print(X_train.shape)
    y = [1.0 ]*10

    pred1 = DIS(X_train)
    pred2 = DIS.predict(X_train)

    v = DIS.trainable_weights# or trainable_variables
    with tf.GradientTape() as tape:
        loss = tf.keras.losses.binary_crossentropy(pred1, y)
        g1 = tape.gradient(loss, DIS.trainable_variables)
        print(g1) # None
        # g2 = tape.gradient(pred2, v)# 报错，tape情况下似乎只有上面那种情况能求梯度
        # A non-persistent GradientTape can only be used to compute one set of gradients (or jacobians)
        # print(g2)
        print('AAAAAAAAAAAAAAAAAA')

    # loss = tf.keras.losses.binary_crossentropy(pred1,y)
    # optimizer = tf.compat.v1.train.AdamOptimizer(0.1)
    # gvs = optimizer.compute_gradients(loss)# 执行失败

if __name__ == '__main__':
    DIS = build_discriminator()
    optimizer = Adam(0.0002, 0.5)
    DIS.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

    train(DIS)
