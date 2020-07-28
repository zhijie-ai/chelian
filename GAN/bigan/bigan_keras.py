#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/4/23 10:37                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

# bigan的实现
# 参考代码:https://github.com/eriklindernoren/Keras-GAN/blob/master/bigan/bigan.py
# 本实现是将z和展开后的X拼接成一个向量
from __future__ import print_function,division

from keras.datasets import mnist
from keras.layers import Input,Dense,Reshape,Flatten,Dropout,multiply,GaussianNoise
from keras.layers import BatchNormalization,Activation,Embedding,ZeroPadding2D
from keras.layers import MaxPooling2D,concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D,Conv2D
from keras.models import Sequential,Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
from keras.utils import plot_model
import keras.backend as K

import matplotlib.pyplot as plt
import numpy as np

class BIGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows,self.img_cols,self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002,0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        #Build the generator
        self.generator = self.build_generator()

        # Build the encoder
        self.encoder = self.build_encoder()

        # The part of the bigan that trains the discriminator and encoder
        self.discriminator.trainable = False

        # Generate image from sampled noise
        z = Input(shape=(self.latent_dim,))
        img_ = self.generator(z)

        # Encode image
        img = Input(shape=self.img_shape)
        z_ = self.encoder(img)

        # Latent -> img is fake,and img -> latent is valid
        fake = self.discriminator([z,img_])
        valid = self.discriminator([z_,img])

        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        self.bigan_generator = Model([z,img],[fake,valid],name='bigan_generator')
        plot_model(self.bigan_generator,to_file='png/bigan_generator.png',show_shapes=True)
        self.bigan_generator.compile(loss=['binary_crossentropy','binary_crossentropy'],
                                     optimizer=optimizer)

    def build_encoder(self):
        model = Sequential(name='encoder')

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.latent_dim))

        img = Input(shape=self.img_shape)
        z = model(img)

        encoder = Model(img, z, name='encoder_model')
        plot_model(encoder, to_file='./png/encoder.png', show_shapes=True)

        return encoder

    def build_generator(self):
        model = Sequential(name='generator')

        model.add(Dense(512,input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape),activation='tanh'))
        model.add(Reshape(self.img_shape))

        z = Input(shape=(self.latent_dim,))
        gen_img= model(z)

        generator = Model(z,gen_img,name='generator_model')
        plot_model(generator, to_file='./png/generator.png', show_shapes=True)
        return generator

    def build_discriminator(self):
        z = Input(shape=(self.latent_dim,))
        img = Input(shape=self.img_shape)
        d_in = concatenate([z,Flatten()(img)])

        model = Dense(1024)(d_in)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        validity = Dense(1,activation='sigmoid')(model)

        discriminator = Model([z,img],validity,name='discriminator')
        plot_model(discriminator, to_file='./png/discriminator.png', show_shapes=True)
        return discriminator

    def train(self,epochs,batch_size=128,sample_interval = 50):

        # Load the dataset
        (X_train,_),(_,_) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32)-127.5)/127.5
        X_train = np.expand_dims(X_train,axis=3)

        # Adversarial ground truths
        valid =np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))

        for epoch in range(epochs):
            # ---------
            # Train Discriminator
            # ---------

            # Sample noise and generate img
            z = np.random.normal(size=(batch_size,self.latent_dim))
            imgs_ = self.generator.predict(z)

            # Select a random batch of imgas and encode
            idx = np.random.randint(0,X_train.shape[0],batch_size)
            imgs = X_train[idx]
            z_ = self.encoder.predict(imgs)

            # Train the discriminator (img -> z is valid, z -> img is fake)
            d_loss_real = self.discriminator.train_on_batch([z_,imgs],valid)
            d_loss_fake = self.discriminator.train_on_batch([z,imgs_],fake)
            d_loss = 0.5*np.add(d_loss_real,d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator(z -> img is valid and img -> z is invalid
            g_loss = self.bigan_generator.train_on_batch([z,imgs],[valid,fake])

            # Plot the progress
            print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0],
                                                                 100 * d_loss[1], g_loss[0]))

            # If at save interval => save generated images samples
            if epoch %sample_interval ==0:

                self.sample_interval(epoch)


    def sample_interval(self,epoch):
            r,c = 5,5
            z = np.random.normal(size=(25,self.latent_dim))
            gen_imgs = self.generator.predict(z)

            gen_imgs = 0.5*gen_imgs + 0.5

            fig,axs = plt.subplots(r,c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(gen_imgs[cnt,:,:,0],cmap='gray')
                    axs[i,j].axis('off')
                    cnt+=1
            fig.savefig('images/mnist_%d.png'%epoch)
            plt.close()


if __name__ == '__main__':
    bigan = BIGAN()
    bigan.train(epochs=40000,batch_size=32,sample_interval=400)