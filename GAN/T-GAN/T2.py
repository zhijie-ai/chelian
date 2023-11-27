#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Tencent Inc.
# Author: ranchodzu (ranchodzu@tencent.com)
# TIME: 2023/1/18 22:59

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model,Sequential
import tensorflow as tf
import numpy as np

class Encoder(Layer):
    def __init__(self, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense = Dense(6)
        self.bn = BatchNormalization()

    def call(self, x):
        x = self.dense(x) # 如果用这个，en.trainable_weights有值，用下面那个，为空
        x = self.bn(x)
        # x = Dense(6,trainable=True)(x)
        return x

def build_discriminator():
    model = Sequential()

    model.add(Dense(10, input_shape=(10,)))
    model.add(Dense(1, activation='sigmoid'))

    img = Input(shape=(10,))
    validity = model(img)
    # model = Model(img, validity)
    model.summary()
    return model

if __name__ == '__main__':
    en = Encoder()
    y = en(tf.ones(shape=(3, 4)))
    print(en.weights)
    print(en.trainable_variables)
    print(en.trainable_weights)
    # dis = build_discriminator()

