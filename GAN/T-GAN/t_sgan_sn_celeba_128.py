#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Tencent Inc.
# Author: ranchodzu (ranchodzu@tencent.com)
# TIME: 2023/1/5 11:30

import numpy as np
from scipy import misc
import glob
import imageio
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
import os
from tqdm import tqdm
import time
from skimage.transform import resize
import matplotlib.pyplot as plt

if not os.path.exists('samples'):
    os.mkdir('samples')

imgs = glob.glob('/apdcephfs_cq2/share_919031/ranchodzu/GAN/images/img_align_celeba/*.jpg')
np.random.shuffle(imgs)
img_dim = 128
z_dim = 100
l2_w = 0.06

def plot_figure(data, data_t, name):
    plt.figure()
    plt.plot(range(len(data)), data, label='D loss', color='red')
    plt.plot(range(len(data_t)), data_t, label='G loss', color='blue')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(name)
    plt.close()

def imread(f):
    x = plt.imread(f)
    print(x.shape)
    x = resize(x, (img_dim, img_dim))
    return x.astype(np.float32)/ 255*2 - 1

def data_generator(batch_size=1024):
    X = []
    while True:
        np.random.shuffle(imgs)
        for f in imgs:
            X.append(imread(f))
            if len(X) == batch_size:
                X = np.array(X)
                yield X
                X = []

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

# 编码器, 输入一张图片，输出该图片对应的特征向量
x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in

x = Conv2D(img_dim, (5,5), strides=(2, 2), padding='same',
           kernel_constraint=spectral_normalization, kernel_regularizer=l2(l2_w))(x)
x = LeakyReLU()(x)

for i in range(3):
    x = Conv2D(img_dim * 2**(i+1), (5, 5), strides=(2,2), padding='same',
               kernel_constraint=spectral_normalization, kernel_regularizer=l2(l2_w))(x)
    x = BatchNormalization(gamma_constraint=spectral_normalization)(x)
    x = LeakyReLU()(x)  # (?, 8, 8, 1024)

x = GlobalAveragePooling2D()(x)  # (?, 1024)

e_model = Model(x_in, x)
e_model.summary()

# 判别器, 输入一个向量，输出一个实数
z_in = Input(shape=(K.int_shape(x)[-1], ))
z = z_in

z = Dense(512, kernel_constraint=spectral_normalization, kernel_regularizer=l2(l2_w))(z)
z = LeakyReLU()(z)
z = Dense(1, use_bias=False,
          kernel_constraint=spectral_normalization,
          activation='sigmoid', kernel_regularizer=l2(l2_w))(z)
d_model = Model(z_in, z)
d_model.summary()

#生成器, 输入一个向量，输出一张图片
z_in = Input(shape=(z_dim, ))
z = z_in

z = Dense(4 * 4 * img_dim * 8)(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Reshape((4, 4, img_dim* 8))(z)

for i in range(4):
    z = Conv2DTranspose(img_dim * 4 //2**i, (5, 5), strides=(2,2), padding='same')(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)

z = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same')(z)
z = Activation('tanh')(z)

g_model = Model(z_in, z)
g_model.summary()


# 整合模型 (训练判别器)
x_in = Input(shape=(img_dim, img_dim, 3))
z_in = Input(shape=(z_dim, ))
g_model.trainable = False

x_fake = g_model(z_in)
x_real_encoded = e_model(x_in)
x_fake_encoded = e_model(x_fake)
x_real_fake = Subtract()([x_real_encoded, x_fake_encoded])
x_fake_real = Subtract()([x_fake_encoded, x_real_encoded])
x_real_fake_score = d_model(x_real_fake)
x_fake_real_score = d_model(x_fake_real)

d_train_model = Model([x_in, z_in], [x_real_fake_score, x_fake_real_score])
d_loss = K.mean(-K.log(x_real_fake_score+1e-9) -K.log(1 - x_fake_real_score+1e-9))
d_train_model.add_loss(d_loss)
d_train_model.compile(optimizer=Adam(2e-4, 0.5))

# 整合模型 (训练生成器)
g_model.trainable = True
d_model.trainable = False
e_model.trainable = False

x_fake = g_model(z_in)
x_real_encoded = e_model(x_in)
x_fake_encoded = e_model(x_fake)
x_real_fake = Subtract()([x_real_encoded, x_fake_encoded])
x_fake_real = Subtract()([x_fake_encoded, x_real_encoded])
x_real_fake_score = d_model(x_real_fake)
x_fake_real_score = d_model(x_fake_real)

g_train_model = Model([x_in, z_in], [x_real_fake_score, x_fake_real_score])
g_loss = K.mean(- K.log(1-x_real_fake_score + 1e-9)- K.log(x_fake_real_score+1e-9))
g_train_model.add_loss(g_loss)
g_train_model.compile(optimizer=Adam(2e-4, 0.5))

# 检测模型结构
d_train_model.summary()
g_train_model.summary()

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


iters_per_sample = 100
total_iter = 10000
batch_size=64
img_generator = data_generator(batch_size)

print('model training starting.....')
t1 = time.time()
print('============', time.strftime('%Y-%m-%d %H:%M:%S'))
D_loss = []
G_loss = []
for i in tqdm(range(total_iter)):
    for j in range(1):
        z_sample = np.random.randn(batch_size, z_dim)
        d_loss = d_train_model.train_on_batch([next(img_generator), z_sample], None)
        D_loss.append(d_loss)
    for j in range(1):
        z_sample = np.random.randn(batch_size, z_dim)
        g_loss = g_train_model.train_on_batch([next(img_generator), z_sample], None)
        G_loss.append(g_loss)
        if i % 10 ==0:
            print(
            'iter: %s, d_loss: %s, g_loss: %s' % (i, d_loss, g_loss))
        if i%iters_per_sample ==0:
            sample('samples/test_{}_{}_{}.png'.format(i, total_iter, l2_w))
        g_train_model.save_weights('./g_train_model.h5')
t2 = time.time()
plot_figure(D_loss,G_loss, 'png/gan-loss-{}-{}.png'.format(total_iter,l2_w))
print('============', time.strftime('%Y-%m-%d %H:%M:%S'))
print('model training success..... time cost:{} h'.format((t2-t1)/(60*60)))
