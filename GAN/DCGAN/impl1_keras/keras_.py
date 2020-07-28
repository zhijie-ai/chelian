#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/6/24 10:46                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
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
from keras.utils import np_utils

x_in = Input(shape=(28,28,1))
model = Conv2D(32,kernel_size=3,strides=2,padding='same')(x_in)
model = LeakyReLU(alpha=0.2)(model)
model = Dropout(0.25)(model)
model = Conv2D(64,kernel_size=3,strides=2,padding='same')(model)
model = ZeroPadding2D(padding=((0,1),(0,1)))(model)
model = BatchNormalization(momentum=0.8)(model)
model = LeakyReLU(alpha=0.2)(model)
model = Dropout(0.25)(model)
model = Conv2D(128,kernel_size=3,strides=2,padding='same')(model)
model = BatchNormalization(momentum=0.8)(model)
model = LeakyReLU(alpha=0.2)(model)
model = Dropout(0.25)(model)
model = Conv2D(256,kernel_size=3,strides=1,padding='same')(model)
model = Flatten()(model)
model = Dense(256,activation='sigmoid')(model)
model = Dense(128,activation='sigmoid')(model)
model = Dense(10,activation='softmax')(model)

model = Model(x_in,model)


(X_train,y_train),(_,_) = mnist.load_data()
X_train = np.expand_dims(X_train,axis=3)
# y_train  = np_utils.to_categorical(y_train, num_classes=10)
print(X_train.shape,y_train.shape)

model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(0.2))
model.fit(X_train,y_train,epochs=2)
