#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/4/17 11:19                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# https://blog.csdn.net/nima1994/article/details/83620725
# keras版的实现

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows= 28
        self.img_cols= 28
        self.channels = 1
        self.img_shape = (self.img_rows,self.img_cols,self.channels)
        self.latent_dim =100

        optimizer = Adam(0.0002,0.5)

        base_generator = self.build_generator()
        base_discriminator = self.build_discriminator()
        print('AAAAAAAAAAA',base_discriminator.outputs)

        self.generator = Model(inputs=base_generator.inputs,
                               outputs=base_generator.outputs,name='generator')
        plot_model(self.generator, to_file='png/generator.png', show_shapes=True)

        self.discriminator = Model(inputs = base_discriminator.inputs,
                                   outputs= base_discriminator.outputs,name='discriminator')
        plot_model(self.discriminator, to_file='png/discriminator.png', show_shapes=True)

        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        frozen_D = Model(inputs=base_discriminator.inputs,
                         outputs=base_discriminator.outputs,name='frozen_D')
        frozen_D.trainable=False
        z= Input(shape=(self.latent_dim,))
        img = self.generator(z)
        valid = frozen_D(img)
        self.combined = Model(z,valid,name='combined')
        plot_model(self.combined,to_file='png/combined.png',show_shapes=True)

        self.combined.compile(loss='binary_crossentropy',optimizer=optimizer)

    def build_generator(self):
        model = Sequential(name='build_generator')
        model.add(Dense(128*7*7,activation='relu',input_dim=(self.latent_dim)))
        model.add(Reshape((7,7,128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128,kernel_size=3,padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(UpSampling2D())
        model.add(Conv2D(64,kernel_size=3,padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(Conv2D(self.channels,kernel_size=3,padding='same'))
        model.add(Activation('tanh'))
        plot_model(model,to_file='png/build_generator.png',show_shapes=True)

        return model

    def build_discriminator(self):
        model = Sequential(name='build_discriminator')
        model.add(Conv2D(32,kernel_size=3,strides=2,input_shape=self.img_shape,padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64,kernel_size=3,strides=2,padding='same'))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128,kernel_size=3,strides=2,padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256,kernel_size=3,strides=1,padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1,activation='sigmoid'))
        plot_model(model,to_file='png/build_discriminator.png',show_shapes=True)

        return model

    def train(self,epochs,batch_size,save_interval,log_interval):
        # Load the dataset
        (X_train,_),(_,_) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 -1
        X_train = np.expand_dims(X_train,axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))

        logs = []

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0,X_train.shape[0],batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0,1,(batch_size,self.latent_dim))
            gen_images = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs,valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_images,fake)
            d_loss = 0.5*np.add(d_loss_real,d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real
            g_loss = self.combined.train_on_batch(noise,valid)

            if epoch % log_interval ==0:
                logs.append([epoch,d_loss[0],d_loss[1],g_loss])

            if epoch %save_interval ==0:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                      (epoch, d_loss[0], 100 * d_loss[1], g_loss))
                self.save_imgs(epoch)
        self.showlogs(logs)

    def showlogs(self,logs):
        logs = np.array(logs)
        names = ['d_loss','d_acc','g_loss']
        for i in range(3):
            plt.subplot(2,2,i+1)
            plt.plot(logs[:,0],logs[:,i+1])
            plt.xlabel('epoch')
            plt.ylabel(names[i])

        plt.tight_layout()
        plt.show()

    def save_imgs(self,epoch):
        r,c = 5,5
        noise = np.random.normal(0,1,(r*c,self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        print(gen_imgs.max(),gen_imgs.min())

        # Rescale images 0 - 1
        gen_imgs = 0.5*gen_imgs+0.5

        fig,axs = plt.subplots(r,c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0],cmap='gray')
                axs[i,j].axis('off')
                cnt+=1

        fig.savefig('images/mnist_%d.png' %epoch)
        plt.close()

if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000,batch_size=32,save_interval=50,log_interval=10)