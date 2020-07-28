#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/7/2 15:50                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# https://github.com/mafda/generative_adversarial_networks_101/blob/master/src/mnist/04_CCGAN_MNIST.ipynb

import numpy as np

import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization
from keras.layers import Input, Flatten, Embedding, multiply, Dropout
from keras.optimizers import Adam
from keras import initializers

# from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Concatenate, GaussianNoise
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
from keras.utils import plot_model

# from scipy.misc import imresize
from skimage.transform import resize

(X_train, y_train), (X_test, y_test) = mnist.load_data()

fig = plt.figure()
for i in range(10):
    plt.subplot(2, 5, i + 1)
    x_y = X_train[y_train == i]
    plt.imshow(x_y[0], cmap='gray', interpolation='none')
    plt.title("Class %d" % (i))
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()

print('X_train.shape', X_train.shape)
print('y_train.shape', y_train.shape)

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

# the generator is using tanh activation, for which we need to preprocess
# the image data into the range between -1 and 1.

X_train = np.float32(X_train)
X_train = (X_train / 255 - 0.5) * 2
X_train = np.clip(X_train, -1, 1)

# y to categorical
num_classes = 10
y_train = to_categorical(y_train, num_classes=num_classes+1)

print('X_train reshape:', X_train.shape)
print('y_train reshape:', y_train.shape)



X_train.resize(X_train.shape[0], 32, 32, 1)
print(type(X_train))
print('X_train reshape:', X_train.shape)


# Generator
gf = 32
k = 4
s = 2

# imagem shape 28x28x1
img_shape = X_train[0].shape

# Generator input
img_g = Input(shape=(img_shape))

# Downsampling
d1 = Conv2D(gf, kernel_size=k, strides=s, padding='same')(img_g)
d1 = LeakyReLU(alpha=0.2)(d1)

d2 = Conv2D(gf*2, kernel_size=k, strides=s, padding='same')(d1)
d2 = LeakyReLU(alpha=0.2)(d2)
d2 = BatchNormalization(momentum=0.8)(d2)

d3 = Conv2D(gf*4, kernel_size=k, strides=s, padding='same')(d2)
d3 = LeakyReLU(alpha=0.2)(d3)
d3 = BatchNormalization(momentum=0.8)(d3)

d4 = Conv2D(gf*8, kernel_size=k, strides=s, padding='same')(d3)
d4 = LeakyReLU(alpha=0.2)(d4)
d4 = BatchNormalization(momentum=0.8)(d4)
print('==========d4.shape',d4.shape)#?, 2, 2, 256)

# Upsampling
u1 = UpSampling2D(size=2)(d4)
u1 = Conv2D(gf*4, kernel_size=k, strides=1, padding='same', activation='relu')(u1)
u1 = BatchNormalization(momentum=0.8)(u1)#(4,4)

u2 = Concatenate()([u1, d3])
u2 = UpSampling2D(size=2)(u2)
u2 = Conv2D(gf*2, kernel_size=k, strides=1, padding='same', activation='relu')(u2)
u2 = BatchNormalization(momentum=0.8)(u2)#(8,8)

u3 = Concatenate()([u2, d2])
u3 = UpSampling2D(size=2)(u3)
u3 = Conv2D(gf, kernel_size=k, strides=1, padding='same', activation='relu')(u3)
u3 = BatchNormalization(momentum=0.8)(u3)

u4 = Concatenate()([u3, d1])
u4 = UpSampling2D(size=2)(u4)
u4 = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
print('===========u4.shape',u4.shape)

generator = Model(img_g, u4,name='auto-encoder')
plot_model(generator,show_shapes=True,to_file='png/generator.png')

# Discriminator
k = 4

discriminator = Sequential(name='discriminator')
discriminator.add(Conv2D(64, kernel_size=k, strides=2, padding='same', input_shape=img_shape))
discriminator.add(LeakyReLU(alpha=0.8))
discriminator.add(Conv2D(128, kernel_size=k, strides=2, padding='same'))
discriminator.add(LeakyReLU(alpha=0.2))
# discriminator.add(InstanceNormalization())
discriminator.add(Conv2D(256, kernel_size=k, strides=2, padding='same'))
discriminator.add(LeakyReLU(alpha=0.2))
# discriminator.add(InstanceNormalization())

img_d = Input(shape=img_shape)
features = discriminator(img_d)#(4,4,256)
plot_model(discriminator,show_shapes=True,to_file='png/discriminator.png')

validity = Conv2D(1, kernel_size=k, strides=1, padding='same')(features)#(4,4,1)
# validity = Flatten()(validity)
# validity = Dense(1, activation='sigmoid')(validity)

label = Flatten()(features)
label = Dense(num_classes+1, activation="softmax")(label)#(11,)

discriminator = Model(img_d, [validity, label],name='discriminator')
plot_model(discriminator,show_shapes=True,to_file='png/discriminator2.png')

# Compile model
optimizer = Adam(lr=0.0002, beta_1=0.5)

discriminator.compile(optimizer=optimizer, loss=['mse', 'categorical_crossentropy'],
                      loss_weights=[0.5, 0.5], metrics=['accuracy'])

# Combined network
masked_img = Input(shape=(img_shape))
gen_img = generator(masked_img)

# For the combined model we will only train the generator
discriminator.trainable = False

validity, _ = discriminator(gen_img)

d_g = Model(masked_img, validity)

d_g.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

def mask_randomly(imgs, mask_width=10, mask_height=10):
    y1 = np.random.randint(0, imgs.shape[1] - mask_height, imgs.shape[0])
    y2 = y1 + mask_height
    x1 = np.random.randint(0, imgs.shape[2] - mask_width, imgs.shape[0])
    x2 = x1 + mask_width

    masked_imgs = np.empty_like(imgs)
    for i, img in enumerate(imgs):
        masked_img = img.copy()
        _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i],
        masked_img[_y1:_y2, _x1:_x2, :] = 0
        masked_imgs[i] = masked_img

    return masked_imgs

# Fit model
epochs = 100
batch_size = 4
smooth = 0.1

real = np.ones((batch_size, 4, 4, 1))
real = real * (1 - smooth)
fake = np.zeros((batch_size, 4, 4, 1))

fake_labels = to_categorical(np.full((batch_size, 1), num_classes), num_classes=num_classes + 1)

d_loss = []
d_g_loss = []

for e in range(epochs + 1):
    for i in range(len(X_train) // batch_size):
        # Train Discriminator weights
        discriminator.trainable = True

        # Real samples
        img_real = X_train[i * batch_size:(i + 1) * batch_size]
        real_labels = y_train[i * batch_size:(i + 1) * batch_size]

        d_loss_real = discriminator.train_on_batch(x=img_real, y=[real, real_labels])

        # Fake Samples
        masked_imgs = mask_randomly(img_real)
        gen_imgs = generator.predict(masked_imgs)

        d_loss_fake = discriminator.train_on_batch(x=gen_imgs, y=[fake, fake_labels])

        # Discriminator loss
        d_loss_batch = 0.5 * (d_loss_real[0] + d_loss_fake[0])

        # Train Generator weights
        discriminator.trainable = False

        d_g_loss_batch = d_g.train_on_batch(x=img_real, y=real)#这个地方应该是masked_imgs

        print(
            'epoch = %d/%d, batch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (
            e + 1, epochs, i, len(X_train) // batch_size, d_loss_batch, d_g_loss_batch[0]),
            100 * ' ',
            end='\r'
        )

    d_loss.append(d_loss_batch)
    d_g_loss.append(d_g_loss_batch[0])
    print('epoch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (e + 1, epochs, d_loss[-1], d_g_loss[-1]), 100 * ' ')

    if e % 10 == 0:
        samples = 5
        idx = np.random.randint(0, X_train.shape[0], samples)
        masked_imgs = mask_randomly(X_train[idx])
        x_fake = generator.predict(masked_imgs)
        print('-------------',x_fake.max(),x_fake.min())

        for k in range(samples):
            # plot masked
            plt.subplot(2, 5, k + 1)
            plt.imshow(masked_imgs[k].reshape(32, 32), cmap='gray')
            plt.xticks([])
            plt.yticks([])

            # plot recontructed
            plt.subplot(2, 5, k + 6)
            plt.imshow(x_fake[k].reshape(32, 32), cmap='gray')
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.show()

# Evaluate model
plt.plot(d_loss)
plt.plot(d_g_loss)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Discriminator', 'Adversarial'], loc='center right')
plt.show()