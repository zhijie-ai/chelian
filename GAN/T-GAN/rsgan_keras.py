#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Tencent Inc.
# Author: ranchodzu (ranchodzu@tencent.com)
# TIME: 2023/1/20 09:40
import numpy as np
from scipy import misc
import imageio
import glob
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import os
from tqdm import tqdm
import time
from skimage.transform import resize
import matplotlib.pyplot as plt
from options import get_options
# https://github.com/zhijie-ai/TensorFlow-2.x-Tutorials/blob/master/13-DCGAN/main.py

if not os.path.exists('samples'):
    os.mkdir('samples')
if not os.path.exists('png'):
    os.mkdir('png')
if not os.path.exists('model'):
    os.mkdir('model')

# rsgan
imgs = glob.glob('/apdcephfs_cq2/share_919031/ranchodzu/GAN/images/img_align_celeba/*.jpg')
# imgs = glob.glob('../../images/img_align_celeba/*.jpg')
np.random.shuffle(imgs)
img_dim = 128
z_dim = 100

def imread(f):
    x = plt.imread(f)
    x = resize(x, (img_dim, img_dim))
    return x.astype(np.float32)/ 255*2 - 1

def data_generator(batch_size=1024):
    X = []
    while True:
        np.random.shuffle(imgs)
        for f in imgs:
            X.append(imread(f))
            if len(X) == batch_size:
                z_sample = np.random.randn(batch_size, z_dim)
                X = np.array(X)
                yield [X,z_sample]
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

def sample(model, path):
    n = 3
    figure = np.zeros((img_dim*n, img_dim * n, 3))
    for i in range(n):
        for j in range(n):
            z_sample = np.random.randn(1, z_dim)
            x_sample = model.predict(z_sample)
            digit = x_sample[0]
            figure[i*img_dim:(i+1)*img_dim, j * img_dim:(j+1)*img_dim] = digit
    figure = (figure+1)/2*255
    figure = np.round(figure, 0).astype(np.uint8)
    imageio.imwrite(path, figure)

def plot_figure(data, data_t, name):
    plt.figure()
    plt.plot(range(len(data)), data, label='D loss', color='red')
    plt.plot(range(len(data_t)), data_t, label='G loss', color='blue')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(name)
    plt.close()

class TGAN():
    def __init__(self, args):
        self.l2_w = args.l2_w
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.iters_per_sample = args.iters_per_sample
        self.d_lr = args.d_lr
        self.g_lr = args.g_lr
        self.mode = 'rsgan'
        self.key_word = '{}_{}_{}_{}_{}_{}'.format(self.mode, self.epochs, self.d_lr,
                                                   self.g_lr, self.l2_w, self.batch_size)
        self.discriminator = self.build_discriminator()

        # Build the generator
        self.generator = self.build_generator()

    def __str__(self):
        dit = self.__dict__
        show = ['l2_w', 'epochs', 'iters_per_sample', 'batch_size', 'd_lr', 'key_word', 'g_lr']
        dict = {key: val for key, val in dit.items() if key in show}
        return str(dict)


    def build_discriminator(self):
        # 编码器
        dis_model = Sequential()
        dis_model.add(Conv2D(img_dim, (5, 5), strides=(2, 2), padding='same',
                            input_shape=(img_dim,img_dim, 3),
                             kernel_regularizer=l2(self.l2_w)))
        dis_model.add(LeakyReLU())
        # dis_model.add(Activation('tanh'))
        for i in range(3):
            dis_model.add(Conv2D(img_dim * 2 ** (i + 1), (5, 5), strides=(2, 2), padding='same',
                                kernel_regularizer=l2(self.l2_w)))
            dis_model.add(BatchNormalization())
            dis_model.add(LeakyReLU())
        dis_model.add(GlobalAveragePooling2D())

        dis_model.add(Dense(512, kernel_regularizer=l2(self.l2_w), kernel_constraint=spectral_normalization))
        dis_model.add(LeakyReLU())
        dis_model.add(Dense(256,  kernel_regularizer=l2(self.l2_w)))
        dis_model.add(LeakyReLU())
        dis_model.add(Dense(128,  kernel_regularizer=l2(self.l2_w), kernel_constraint=spectral_normalization))
        dis_model.add(LeakyReLU())
        dis_model.add(Dense(64,  kernel_regularizer=l2(self.l2_w)))
        dis_model.add(LeakyReLU())
        dis_model.add(Dense(32,  kernel_regularizer=l2(self.l2_w), kernel_constraint=spectral_normalization))
        dis_model.add(LeakyReLU())
        dis_model.add(Dense(1, use_bias=False,  kernel_regularizer=l2(self.l2_w)))

        x_real = Input(shape=(img_dim, img_dim, 3))
        x_fake = Input(shape=(img_dim, img_dim, 3))
        x_real_score = dis_model(x_real)
        x_fake_score = dis_model(x_fake)
        model = Model([x_real,x_fake],[x_real_score,x_fake_score])
        return model

    def build_generator(self):
        # 生成器, 输入一个向量，输出一张图片
        z_in = Input(shape=(z_dim,))
        z = z_in

        z = Dense(4 * 4 * img_dim * 8)(z)
        z = BatchNormalization()(z)
        z = Activation('relu')(z)
        z = Reshape((4, 4, img_dim * 8))(z)

        for i in range(4):
            z = Conv2DTranspose(img_dim * 4 // 2 ** i, (5, 5), strides=(2, 2), padding='same',
                                kernel_regularizer=l2(self.l2_w)
                                )(z)
            z = BatchNormalization()(z)
            z = Activation('relu')(z)

        z = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same',  kernel_regularizer=l2(self.l2_w))(z)
        z = Activation('relu')(z)

        g_model = Model(z_in, z)
        return g_model

    def train(self):
        G_loss = []
        D_loss = []
        img_generator = data_generator(self.batch_size)
        d_optimizer = tf.keras.optimizers.SGD(lr=self.d_lr)
        g_optimizer = tf.keras.optimizers.SGD(lr=self.g_lr)
        print('model training starting.....')
        for i in tqdm(range(self.epochs)):
            x, z = next(img_generator)
            # train D
            with tf.GradientTape() as d:
                x_gen = self.generator(z)
                x_real_score, x_fake_score = self.discriminator([x, x_gen])
                d_loss = self.d_loss_fn(x_real_score, x_fake_score)
            d_grads = d.gradient(d_loss, self.discriminator.trainable_weights)
            d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))
            D_loss.append(d_loss)

            # train G
            with tf.GradientTape() as g:
                x_gen = self.generator(z)
                x_real_score, x_fake_score = self.discriminator([x, x_gen])
                g_loss = self.g_loss_fn(x_real_score, x_fake_score)
            g_grads = g.gradient(g_loss, self.generator.trainable_weights)
            g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))
            G_loss.append(g_loss)
            print('------------------------------------------', 'x_real_score',x_real_score[:6],
                  'x_fake_score',x_fake_score[:6],
                  'd_loss', d_loss, 'd_grads', d_grads[0][0][0][0][:6], 'g_loss', g_loss, 'g_grads', g_grads[0][0][:6])
            if i % 10 == 0:
                print(
                    'iter: %s, d_loss: %s, g_loss: %s' % (i, d_loss, g_loss))
            if i % self.iters_per_sample == 0:
                sample(self.generator, 'samples/{}_{}.png'.format(self.key_word, i))
            self.generator.save_weights('model/{}_g_train_model.h5'.format(self.key_word))

        plot_figure(D_loss, G_loss, 'png/{}.png'.format(self.key_word))

    def d_loss_fn(self, x_real_score, x_fake_score):
        loss = -K.mean(K.log(K.sigmoid(x_real_score - x_fake_score)+1e-9))# log里面的值>0
        return loss

    def g_loss_fn(self, x_real_score, x_fake_score):
        loss = -K.mean(K.log(K.sigmoid(x_fake_score - x_real_score)+1e-9))
        return loss


# 训练是__main__=='tensorflow.keras.layers'
print('model training starting.....')
args = get_options()

t1 = time.time()
print('============', time.strftime('%Y-%m-%d %H:%M:%S'))
tgan = TGAN(args)
print('>>>>>>>>>>>>>>>>>>>>>>>>',tgan,'<<<<<<<<<<<<<<<<<<<<<<<<<<<')
tgan.train()
t2 = time.time()
print('============', time.strftime('%Y-%m-%d %H:%M:%S'))
print('model training success..... time cost:{} h'.format((t2 - t1) / (60 * 60)))