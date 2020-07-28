#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/6/25 10:39                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import numpy as np
from keras.datasets import mnist
from  keras.utils import to_categorical

(X_train, y_train), (_, _) = mnist.load_data()
X_train = X_train.reshape(-1,784)
y_train = to_categorical(y_train,10)

np.random.seed(1)
model = Sequential()

model.add(Dense(input_dim=X_train.shape[1],
                output_dim=50,
                init='uniform',
                activation='tanh'))

model.add(Dense(input_dim=50,
                output_dim=50,
                init='uniform',
                activation='tanh'))

model.add(Dense(input_dim=50,
                output_dim=10,
                init='uniform',
                activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
model.metrics_names.append('global_info_loss')
import keras.backend as K
model.metrics_tensors.append(K.sum(X_train))
print(model.metrics_names)
model.fit(X_train,
          y_train,
          nb_epoch=4,
          batch_size=300,
          verbose=1)
print(model.metrics_names)

# for epoch in range(50):
#     idx = np.random.randint(0, X_train.shape[0], 64)
#     imgs = X_train[idx]
#
#     loss = model.train_on_batch(imgs,y_train[idx])
#     print(loss,model.metrics_names)
