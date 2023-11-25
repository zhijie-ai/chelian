#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Tencent Inc.
# Author: ranchodzu (ranchodzu@tencent.com)
# TIME: 2023/1/11 22:09

# 这种方法不太可取，每一个layer都要在init方法中定义，否则无法读取到权重。
import numpy as np
from scipy import misc
import glob
import imageio
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import os
from tqdm import tqdm
import time
from skimage.transform import resize
import matplotlib.pyplot as plt

if not os.path.exists('samples'):
    os.mkdir('samples')

imgs = glob.glob('/apdcephfs_cq2/share_919031/ranchodzu/GAN/images/img_align_celeba/*.jpg')
imgs = glob.glob('../../images/img_align_celeba/*.jpg')
np.random.shuffle(imgs)
img_dim = 128
z_dim = 100

def spectral_norm(w, r=5):
    w_shape= K.int_shape(w)
    in_dim = np.prod(w_shape[:-1]).astype(int)
    out_dim = w_shape[-1]
    w = K.reshape(w,(in_dim, out_dim))
    u = K.ones((1, in_dim))
    for i in range(r):
        v = K.l2_normalize(K.dot(u,w))
        u = K.l2_normalize(K.dot(v, K.transpose(w)))
    return K.sum(K.dot(K.dot(u, w), K.transpose(v)))

def spectral_normalization(w):
    return w/ spectral_norm(w)

# 采样函数
def sample(path):
    n = 3
    figure = np.zeros((img_dim*n, img_dim * n, 3))
    for i in range(n):
        for j in range(n):
            z_sample = np.random.randn(1, z_dim)
            x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i*img_dim:(i+1)*img_dim, j * img_dim:(j+1)*img_dim] = digit
    figure = (figure+1)/2*255
    figure = np.round(figure, 0).astype(int)
    imageio.imwrite(path, figure)

def data_generator(batch_size=1024):
    X = []
    while True:
        np.random.shuffle(imgs)
        for f in imgs:
            X.append(imageio.imread(f))
            if len(X) == batch_size:
                z_sample = np.random.randn(batch_size, z_dim)
                X = np.array(X)
                yield [X,z_sample]
                X = []
                
class Conv_EN(Layer):
    def __init__(self, kernels, name='Conv_EN', **kwargs):
        super(Conv_EN, self).__init__(**kwargs)
        self.kernels = kernels
        self.conv = Conv2D(kernels, (5, 5), strides=(2, 2), padding='same',
                   kernel_constraint=spectral_normalization)
        self.bn = BatchNormalization(gamma_constraint=spectral_normalization)
        self.LeakyReLU = LeakyReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.LeakyReLU(x)
        return x

    def get_config(self):
        base_config = super(Conv_EN, self).get_config()  # 父类config字典
        base_config['kernels'] = self.kernels  # 继承类config字典，__init__传入参数arg
        return base_config  # 返回组装后的字典

class Conv_GEN(Layer):
    def __init__(self, kernels, name='Conv_GEN', **kwargs):
        super(Conv_GEN, self).__init__(**kwargs)
        self.kernels = kernels
        self.conv_t = Conv2DTranspose(kernels, (5, 5), strides=(2, 2), padding='same',
                   kernel_constraint=spectral_normalization)
        self.bn = BatchNormalization(gamma_constraint=spectral_normalization)
        self.Activation = Activation('relu')

    def call(self, x):
        x = self.conv_t(x)
        x = self.bn(x)
        x = self.Activation(x)
        return x

    def get_config(self):
        base_config = super(Conv_GEN, self).get_config()  # 父类config字典
        base_config['kernels'] = self.kernels  # 继承类config字典，__init__传入参数arg
        return base_config  # 返回组装后的字典
    
class Encoder(Layer):
    def __init__(self, name='encoder', **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.conv2d_128 = Conv_EN(128)
        self.conv2d_256 = Conv_EN(256)
        self.conv2d_512 = Conv_EN(512)
        self.conv2d_1024 = Conv_EN(1024)
        self.GlobalAveragePooling2D = GlobalAveragePooling2D()

    def call(self, x):
        x = self.conv2d_128(x)
        x = self.conv2d_256(x)
        x = self.conv2d_512(x)
        x = self.conv2d_1024(x)
        x = self.GlobalAveragePooling2D(x)
        return x

    def get_config(self):
        base_config = super(Conv_GEN, self).get_config()  # 父类config字典
        return base_config  # 返回组装后的字典

class DIS(Layer):
    def __init__(self, name='DIS', **kwargs):
        super(DIS, self).__init__(**kwargs)
        self.dense = Dense(512, kernel_constraint=spectral_normalization)
        self.LeakyReLU = LeakyReLU()
        self.dense_1 = Dense(1, use_bias=False,
                  kernel_constraint=spectral_normalization,
                  activation='sigmoid')

    def call(self, z):
        z = self.dense(z)
        z = self.LeakyReLU()(z)
        z = self.dense_1(z)
        return z

    def get_config(self):
        base_config = super(DIS, self).get_config()  # 父类config字典
        return base_config  # 返回组装后的字典

class Discriminator(Layer):
    def __init__(self, name='DIS', **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.encoder = Encoder()
        self.dis = DIS()

    def call(self, x_real, x_fake):
        x_real_encoded = self.encoder(x_real)
        x_fake_encoded = self.encoder(x_fake)
        x_real_fake = Subtract()([x_real_encoded, x_fake_encoded])
        x_fake_real = Subtract()([x_fake_encoded, x_real_encoded])
        x_real_fake_score = self.dis(x_real_fake)
        x_fake_real_score = self.dis(x_fake_real)
        return x_real_fake_score, x_fake_real_score

    def get_config(self):
        base_config = super(Discriminator, self).get_config()  # 父类config字典
        return base_config  # 返回组装后的字典

class Generator(Layer):
    def __init__(self, name='gen', **kwargs):
        super(Generator, self).__init__(name = name, **kwargs)
        self.dense = Dense(4 * 4 * img_dim * 8)
        self.bn = BatchNormalization()
        self.activation = Activation('relu')
        self.Conv2DTranspose_512 = Conv_GEN(img_dim * 4 // 2 ** 0)
        self.Conv2DTranspose_256 = Conv_GEN(img_dim * 4 // 2 ** 1)
        self.Conv2DTranspose_128 = Conv_GEN(img_dim * 4 // 2 ** 2)
        self.Conv2DTranspose_64 = Conv_GEN(img_dim * 4 // 2 ** 3)
        self.Conv2DTranspose_3 = Conv2DTranspose(3, (5, 5))
        self.tanh = Activation('relu')

    def call(self, z):
        z = self.dense(z)
        z = self.bn(z)
        z = self.activation(z)
        z = Reshape((4, 4, img_dim * 8))(z)
        z = self.Conv2DTranspose_512(z)
        z = self.Conv2DTranspose_256(z)
        z = self.Conv2DTranspose_128(z)
        z = self.Conv2DTranspose_64(z)

        z = self.Conv2DTranspose_3(z)
        z = self.tanh(z)

        return z

    def get_config(self):
        base_config = super(Generator, self).get_config()  # 父类config字典
        return base_config  # 返回组装后的字典

class TGAN(Model):
    def __init__(self, name='tgan', **kwargs):
        super(TGAN, self).__init__(**kwargs)
        self.DIS = Discriminator()
        self.GEN = Generator()

    def call(self, x, z):
        x_fake = self.GEN(z)
        x_real_fake_score, x_fake_real_score = self.DIS(x, x_fake)
        return x_real_fake_score, x_fake_real_score


def d_loss_fn(x_real_fake_score, x_fake_real_score):
    loss = K.mean(-K.log(x_real_fake_score + 1e-9) - K.log(1 - x_fake_real_score + 1e-9))
    return loss

def g_loss_fn(x_real_fake_score, x_fake_real_score):
    loss = K.mean(- K.log(1 - x_real_fake_score + 1e-9) - K.log(x_fake_real_score + 1e-9))
    return loss

if __name__ == '__main__':
    iters_per_sample = 100
    total_iter = 10000
    batch_size = 64
    img_generator = data_generator(batch_size)

    tgan = TGAN()
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    print('model training starting.....')
    t1 = time.time()
    D_loss = []
    G_loss = []
    # for epoch in range(total_iter):
    #     x, z = next(img_generator)
    #     with tf.GradientTape() as tape:
    #         x_real_fake_score, x_fake_real_score = tgan(x, z)
    #         d_loss = d_loss_fn(x_real_fake_score, x_fake_real_score)
    #         g_loss = g_loss_fn(x_real_fake_score, x_fake_real_score)
    #         D_loss.append(d_loss)
    #         G_loss.append(g_loss)
    #
    #     grads = tape.gradient(loss, vae.trainable_weights)
    #     optimizer.apply_gradients(zip(grads, vae.trainable_weights))
    x, z = next(img_generator)
    x_real_fake_score, x_fake_real_score = tgan(x, z)
    print(tgan.trainable_variables)