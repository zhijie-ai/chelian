#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/6/25 11:33                       #
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
from keras.initializers import RandomNormal
from keras.optimizers import Adam
import os

imgs = glob.glob('../../images/img_align_celeba/*.jpg')
np.random.shuffle(imgs)
img_dim = 128
z_dim = 128
num_layers = int(np.log2(img_dim))-3
max_num_channels = img_dim*8
f_size = img_dim//2**(num_layers+1)
batch_size = 20


def imread(f):
    x=misc.imread(f,mode='RGB')
    x=misc.imresize(x,(img_dim,img_dim))
    return x.astype(np.float32)/255*2-1

def data_generator(batch_size=64):
    X=[]
    while True:
        np.random.shuffle(imgs)
        for f in imgs:
            X.append(imread(f))
            if len(X)==batch_size:
                X=np.array(X)
                print(X.shape)
                yield X
                X=[]

data = np.arange(129)
print(data)
def data_generator2(batch_size=64):
    X=[]
    while True:
        for f in data:
            X.append(f)
            if len(X)==batch_size:
                X=np.array(X)
                yield X
                X=[]

img_generator = data_generator2()
for i in range(10):#这里循环的次数相当于原来的epochs * (len(X)//batch_size)
    print(next(img_generator),i)