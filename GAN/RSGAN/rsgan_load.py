#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/7/21 11:46                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
from scipy import misc
import glob
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
import os

if not os.path.exists('samples'):
    os.mkdir('samples')

imgs = imgs = glob.glob('../../images/img_align_celeba/*.jpg')
np.random.shuffle(imgs)

print(misc.imread(imgs[0]).shape)

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

def spectral_norm(w, r=5):
    w_shape = K.int_shape(w)
    in_dim = np.prod(w_shape[:-1]).astype(int)
    out_dim = w_shape[-1]
    w = K.reshape(w, (in_dim, out_dim))
    u = K.ones((1, in_dim))
    for i in range(r):
        v = K.l2_normalize(K.dot(u, w))
        u = K.l2_normalize(K.dot(v, K.transpose(w)))
    return K.sum(K.dot(K.dot(u, w), K.transpose(v)))


def spectral_normalization(w):
    return w / spectral_norm(w)

# 判别器
x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in

x = Conv2D(img_dim,
           (5, 5),
           strides=(2, 2),
           padding='same',
           kernel_constraint=spectral_normalization)(x)
x = LeakyReLU()(x)

for i in range(3):
    x = Conv2D(img_dim * 2**(i + 1),
               (5, 5),
               strides=(2, 2),
               padding='same',
               kernel_constraint=spectral_normalization)(x)
    x = BatchNormalization(gamma_constraint=spectral_normalization)(x)
    x = LeakyReLU()(x)

x = Flatten()(x)
x = Dense(1, use_bias=False,
          kernel_constraint=spectral_normalization)(x)

d_model = Model(x_in, x)

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

g_model = Model(z_in, z)

# 整合模型(训练判别器
x_in = Input(shape=(img_dim,img_dim,3))
z_in = Input(shape=(z_dim,))
g_model.trainable=False

x_fake = g_model(z_in)
x_real_score = d_model(x_in)
x_fake_score = d_model(x_fake)

d_train_model = Model([x_in,z_in],[x_real_score,x_fake_score])
d_loss = K.mean(-K.log(K.sigmoid(x_real_score-x_fake_score)+1e-9))
d_train_model.add_loss(d_loss)
d_train_model.compile(optimizer=Adam(2e-4,0.5))

#整合模型(训练生成器)
g_model.trainable=True
d_model.trainable=False

x_fake = g_model(z_in)
x_real_score = d_model(x_in)
x_fake_score = d_model(x_fake)

g_train_model = Model([x_in,z_in],[x_real_score,x_fake_score])
g_train_model.load_weights('g_train_model.weights')
g_model.summary()

import matplotlib.pyplot as plt

z_sample = np.random.randn(1, z_dim)
x=g_model.predict([z_sample])
plt.imshow(x[0])
plt.show()