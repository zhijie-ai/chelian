#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/7/27 9:36                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import numpy as np
from keras.models import Model
from keras.layers import Dense,Conv2DTranspose,Conv2D,Activation,LeakyReLU,Input, \
    Reshape,Flatten,BatchNormalization
from keras import backend as K
from keras.optimizers import Adam


img_dim = 128
z_dim = 128
num_layers = int(np.log2(img_dim)) - 3
max_num_channels = img_dim * 8
f_size = img_dim // 2**(num_layers + 1)
batch_size = 64


# 判别器
x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in

for i in range(num_layers + 1):
    num_channels = max_num_channels // 2**(num_layers - i)
    x = Conv2D(num_channels,
               (5, 5),
               strides=(2, 2),
               use_bias=False,
               padding='same')(x)
    if i > 0:
        x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

x = Flatten()(x)
x = Dense(1, use_bias=False)(x)

d_model = Model(x_in, x,name='qp_d_model_l1_128')


# 生成器
z_in = Input(shape=(z_dim, ))
z = z_in

z = Dense(f_size**2 * max_num_channels)(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Reshape((f_size, f_size, max_num_channels))(z)

for i in range(num_layers):
    num_channels = max_num_channels // 2**(i + 1)
    z = Conv2DTranspose(num_channels,
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

g_model = Model(z_in, z,name='qp_g_model_l1_128')


# 整合模型（训练判别器）
x_in = Input(shape=(img_dim, img_dim, 3))
z_in = Input(shape=(z_dim, ))
g_model.trainable = False

x_real = x_in
x_fake = g_model(z_in)

x_real_score = d_model(x_real)
x_fake_score = d_model(x_fake)

d_train_model = Model([x_in, z_in],
                      [x_real_score, x_fake_score],name='qp_d_train_model_l1_128')

# T 是 (xr,xf) 的二元函数，但实验表明，取最简单的一元特例 T(xr,xf)≡T(xr) 即可，
# 即 T(xr,xf)−T(xf,xr) 用 T(xr)−T(xf) 就够了，改成二元函数并没有明显提升（但也可能是我没调好）
d_loss = x_real_score - x_fake_score #

d_loss = d_loss[:, 0]
d_norm = 10 * K.mean(K.abs(x_real - x_fake), axis=[1, 2, 3])
d_loss = K.mean(- d_loss + 0.5 * d_loss**2 / d_norm)

d_train_model.add_loss(d_loss)
d_train_model.compile(optimizer=Adam(2e-4, 0.5))

# 整合模型（训练生成器）
g_model.trainable = True
d_model.trainable = False

x_real = x_in
x_fake = g_model(z_in)

x_real_score = d_model(x_real)
x_fake_score = d_model(x_fake)

g_train_model = Model([x_in, z_in],
                      [x_real_score, x_fake_score],name='qp_g_train_model_l1_128')

g_loss = K.mean(x_real_score - x_fake_score)

g_train_model.add_loss(g_loss)
g_train_model.compile(optimizer=Adam(2e-4, 0.5))


# 检查模型结构
g_train_model.load_weights('models/g_train_model_l1_best_model.h5')

import matplotlib.pyplot as plt
z_sample = np.random.randn(1, z_dim)
x=g_model.predict([z_sample])
plt.imshow(x[0])
plt.show()

