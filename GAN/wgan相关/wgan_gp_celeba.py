#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/5/27 15:05                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# https://github.com/bojone/gan/blob/master/keras/wgan_gp_celeba.py
# wgan-gp的keras实现
import numpy as np
from scipy import misc
import glob
import imageio
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
import os
from keras.utils import plot_model

if not os.path.exists('samples'):
    os.mkdir('samples')

imgs = glob.glob('../../images/img_align_celeba/*.jpg')
np.random.shuffle(imgs)

height, width = misc.imread(imgs[0]).shape[:2]
center_height = int((height - width) / 2)
img_dim = 64
z_dim = 100

def imread(f):
    x = misc.imread(f)
    x = x[center_height:center_height + width, :]
    x = misc.imresize(x, (img_dim, img_dim))
    return x.astype(np.float32) / 255 * 2 - 1

def data_generator(batch_size=32):
    X = []
    while True:
        np.random.shuffle(imgs)
        for f in imgs:
            X.append(imread(f))
            if len(X) == batch_size:
                X = np.array(X)
                yield X
                X = []

# 判别器
x_in = Input(shape=(img_dim,img_dim,3))
x = x_in

x = Conv2D(img_dim,
           (5, 5),
           strides=(2, 2),
           padding='same')(x)
x = LeakyReLU()(x)

for i in range(3):
    x = Conv2D(img_dim * 2**(i + 1),
               (5, 5),
               strides=(2, 2),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

x = Flatten()(x)
x = Dense(1, use_bias=False)(x)

d_model = Model(x_in, x)
plot_model(d_model,to_file='png/wgan_gp_celeba_d_model.png',show_shapes=True)

# 生成器
z_in = Input(shape=(z_dim, ))
z = z_in

z = Dense(4 * 4 * img_dim * 8)(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Reshape((4, 4, img_dim * 8))(z)

for i in range(3):
    z = Conv2DTranspose(img_dim * 4 // 2**i,
                        (5, 5),
                        strides=(2, 2),
                        padding='same')(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)

z = Conv2DTranspose(3,
                    (5, 5),
                    strides=(2, 2),
                    padding='same')(z)
z = Activation('tanh')(z)
g_model = Model(z_in,z)
plot_model(g_model,to_file='png/wgan_gp_celeba_g_model.png',show_shapes=True)

# 整合模型（训练判别器）
x_in = Input(shape=(img_dim,img_dim,3))
z_in = Input(shape=(z_dim,))
g_model.trainable=False


def interpolating(x):
    print('===============')
    print(x)
    print(K.shape(x[0]))
    print(K.shape(x[0])[0])
    print(K.ndim(x[0]))
    u = K.random_uniform((K.shape(x[0])[0],) + (1,) * (K.ndim(x[0]) - 1))
    print(K.random_uniform((K.shape(x[0])[0],)))
    print('DDDDDDDDDD',u.shape)
    return x[0] * u + x[1] * (1 - u)

x_fake = g_model(z_in)
x_inter=Lambda(interpolating)([x_in,x_fake])
x_real_score = d_model(x_in)
x_fake_score = d_model(x_fake)
x_inter_score = d_model(x_inter)

grads = K.gradients(x_inter_score,[x_inter])[0]
print('AAAAAAAAAA',K.ndim(grads))
print(list(range(1,K.ndim(grads))))
grad_norms = K.sqrt(K.sum(grads**2,axis=list(range(1,K.ndim(grads))))+1e-9)

d_train_model = Model([x_in,z_in],[x_real_score,x_fake_score,x_inter_score])

d_loss = K.mean(-(x_real_score-x_fake_score))+10*K.mean((grad_norms-1)**2)
d_train_model.add_loss(d_loss)
d_train_model.compile(optimizer=Adam(2e-4,0.5))
plot_model(d_train_model,to_file='png/wgan_gp_celeba_d_train_model.png',show_shapes=True)


#整合模型(训练生成器)
g_model.trainable=True
d_model.trainable=False
x_fake_score = d_model(g_model(z_in))

g_train_model=Model(z_in,x_fake_score)
g_train_model.add_loss(K.mean(-x_fake_score))
g_train_model.compile(optimizer=Adam(2e-4,0.5))
plot_model(d_train_model,to_file='png/wgan_gp_celeba_g_train_model.png',show_shapes=True)

# 检查模型结构
# d_train_model.summary()
# g_train_model.summary()

# 采样函数
def sample(path):
    n = 9
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    for i in range(n):
        for j in range(n):
            z_sample = np.random.randn(1, z_dim)
            x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype(int)
    misc.imsave(path, figure)

iters_per_sample = 100
total_iter = 1000000
batch_size = 32
img_generator = data_generator(batch_size)

for i in range(total_iter):
    for j in range(5):
        z_sample = np.random.randn(batch_size, z_dim)
        # d_loss = d_train_model.train_on_batch([next(img_generator), z_sample], None)
        d_loss = d_train_model.train_on_batch([next(img_generator),z_sample],None)
    for j in range(1):
        z_sample = np.random.randn(batch_size, z_dim)
        g_loss = g_train_model.train_on_batch(z_sample, None)
    if i % 10 == 0:
        print('iter: %s, d_loss: %s, g_loss: %s' % (i, d_loss, g_loss))
    if i % iters_per_sample == 0:
        sample('samples/test_%s.png' % i)
        g_train_model.save_weights('./g_train_model.weights')
