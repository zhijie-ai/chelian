#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/4/16 14:56                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

# 本代码是对bigan-qp的实现
# 文章地址：https://www.jiqizhixin.com/articles/2018-11-27-24
# 代码地址：https://github.com/bojone/gan-qp/blob/master/bigan-qp/bigan-qp-256.py

from scipy import misc
import glob
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
img_dim = 256
z_dim = 256
num_layers = int(np.log2(img_dim)) - 3
max_num_channels = img_dim * 8
f_size = img_dim // 2**(num_layers + 1)
batch_size = 2

def imread(f, mode='gan'):
    x = misc.imread(f, mode='RGB')
    if mode == 'gan':
        x = misc.imresize(x, (img_dim, img_dim))
        return x.astype(np.float32) / 255 * 2 - 1
    elif mode == 'fid':
        x = misc.imresize(x, (299, 299))
        return x.astype(np.float32)

class img_generator:
    """
    图片迭代器,方便重复调用
    """
    def __init__(self,imgs,mode='gan',batch_size=64):
        self.imgs = imgs
        self.batch_size=batch_size
        self.mode=mode
        if len(imgs) %batch_size==0:
            self.steps = len(imgs)//batch_size
        else:
            self.steps = len(imgs)//batch_size +1

    def __len__(self):
        return self.steps

    def __iter__(self):
        X=[]
        while True:
            np.random.shuffle(self.imgs)
            for i ,f in enumerate(self.imgs):
                X.append(imread(f,self.mode))
                if len(X) == self.batch_size or i ==len(self.imgs)-1:
                    X=np.array(X)
                    yield X
                    X=[]


# 编码器1(为了编码,把一张图片编码成256维)
x_in = Input(shape=(img_dim,img_dim,3))
x=x_in

for i in range(num_layers + 1):
    num_channels = max_num_channels // 2**(num_layers - i)
    x = Conv2D(num_channels,
               (5, 5),
               strides=(2, 2),
               padding='same')(x)
    if i > 0:
        x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

x=Flatten()(x)
x=Dense(z_dim)(x)

e_model=Model(x_in,x,name='e_model')
e_model.summary()
plot_model(e_model,to_file='./png/bigan_e_model.png',show_shapes=True)

# 编码器2（为了判别器，把一张图片编码成256维）
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
x = Dense(z_dim, use_bias=False)(x)

te_model = Model(x_in, x,name='te_model')
te_model.summary()
plot_model(te_model,to_file='./png/bigan_te_model.png',show_shapes=True)

#判别器
z_in = Input(shape=(z_dim *2,))
z=z_in

z = Dense(1024,use_bias=False)(z)
z=LeakyReLU(0.2)(z)
z=Dense(1024,use_bias=False)(z)
z=LeakyReLU(0.2)(z)
z=Dense(1,use_bias=False)(z)

td_model=Model(z_in,z,name='td_model') # 输出一个实数
td_model.summary()
plot_model(td_model,to_file='./png/bigan_td_model.png',show_shapes=True)

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
    z = LeakyReLU(0.2)(z)

z = Conv2DTranspose(3,
                    (5, 5),
                    strides=(2, 2),
                    padding='same')(z)
z = Activation('tanh')(z)

g_model = Model(z_in, z,name='g_model')
g_model.summary()
plot_model(g_model,to_file='./png/bigan_g_model.png',show_shapes=True)

# 整合模型(训练判别器)
x_in = Input(shape=(img_dim,img_dim,3),name='x_in')
z_in = Input(shape=(z_dim,),name='z_in')
g_model.trainable=False
e_model.trainable=False

x_real,z_fake = x_in,z_in
x_fake = g_model(z_fake)
z_real = e_model(x_real)#将真实图片编码成一个向量，将图片编码成一个z向量
x_real_encoded=te_model(x_real)#代表图片的编码
x_fake_encoded = te_model(x_fake)
xz_real = Concatenate()([x_real_encoded,z_real])
xz_fake = Concatenate()([x_fake_encoded,z_fake])
xz_real_score = td_model(xz_real)
xz_fake_score = td_model(xz_fake)

d_train_model=Model([x_in,z_in],[xz_real_score,xz_fake_score],name='d_train_model')

d_loss = xz_real_score-xz_fake_score
d_loss = d_loss[:,0]
print('qqqqqqqqqqqqqqqqqq',d_loss.shape)
d_norm = 10*(K.mean(K.abs(x_real-x_fake),axis=[1,2,3])+K.mean(K.abs(z_real-z_fake),axis=1))
d_loss=K.mean(-d_loss+0.5*d_loss**2/d_norm)

d_train_model.add_loss(d_loss)
d_train_model.compile(optimizer=Adam(2e-4,0.5))

# 整合模型（训练生成器）
g_model.trainable = True
e_model.trainable = True #将真实图片编码成一个向量，将图片编码成一个z向量
td_model.trainable = False
te_model.trainable = False #代表图片的编码

x_real,z_fake=x_in,z_in
x_fake = g_model(z_fake)
z_real = e_model(x_real)
z_real_ = Lambda(lambda x:K.stop_gradient(x))(z_real)
x_real_ = g_model(z_real_)
x_fake_ = Lambda(lambda x:K.stop_gradient(x))(x_fake)
z_fake_ = e_model(x_fake_)

x_real_encoded = te_model(x_real)
x_fake_encoded = te_model(x_fake)
xz_real = Concatenate()([x_real_encoded, z_real])
xz_fake = Concatenate()([x_fake_encoded, z_fake])
xz_real_score = td_model(xz_real)
xz_fake_score = td_model(xz_fake)

g_train_model = Model([x_in, z_in],[xz_real_score, xz_fake_score],name='g_train_model')
g_loss = K.mean(xz_real_score - xz_fake_score) + \
         4 * K.mean(K.square(z_fake - z_fake_)) + \
         6 * K.mean(K.square(x_real - x_real_))

g_train_model.add_loss(g_loss)
g_train_model.compile(optimizer=Adam(2e-4,0.5))

g_train_model.metrics_names.append('d_loss')
g_train_model.metrics_tensors.append(K.mean(xz_real_score-xz_fake_score))
g_train_model.metrics_names.append('r_loss')
g_train_model.metrics_tensors.append(4*K.mean(K.square(z_fake-z_fake_))+
                                     6 * K.mean(K.square(x_real - x_real_)))

# 检查模型结构
d_train_model.summary()
g_train_model.summary()
plot_model(d_train_model,to_file='./png/bigan_d_train_model.png',show_shapes=True)
plot_model(g_train_model,to_file='./png/bigan_g_train_model.png',show_shapes=True)


# 采样函数
def sample(path,n=8,z_sample=None):
    figure = np.zeros((img_dim*n,img_dim*n,3))
    if z_sample is None:
        z_sample = np.random.randn(n**2,z_dim)
    for i in range(n):
        for j in range(n):
            z_sample = z_sample[[i*n+j]]
            x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i*img_dim:(i+1)*img_dim,j*img_dim:(j+1)*img_dim]=digit
    figure = (figure+1)/2*255
    figure = np.round(figure,0).astype(int)
    misc.imsave(path,figure)


# 重构采样函数
def sample_ae(path, n=8):
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    for i in range(n):
        for j in range(n):
            if j % 2 == 0:
                x_sample = [imread(np.random.choice(imgs))]
            else:
                z_sample = e_model.predict(np.array(x_sample))
                x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype(int)
    misc.imsave(path, figure)


# 插值采样函数
def sample_inter(path,n=8):
    blank=20
    figure = np.zeros((img_dim * n, img_dim * (n + 2) + blank * 2, 3)) + 1
    for i in range(n):
        x_sample_1 = [imread(np.random.choice(imgs))]
        x_sample_2 = [imread(np.random.choice(imgs))]
        figure[i*img_dim:(i+1)*img_dim,:img_dim] = x_sample_1[0]
        figure[i*img_dim:(i+1)*img_dim,-img_dim:] = x_sample_2[0]
        z_sample_1 = e_model.predict(np.array(x_sample_1))
        z_sample_2 = e_model.predict(np.array(x_sample_2))
        for j in range(n):
            z_sample = (1-j/(n-1.))*z_sample_1+j/(n-1.)*z_sample_2
            x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i*img_dim:(i+1)*img_dim,(j+1)*img_dim+blank:(j+2)*img_dim+blank]=digit
    figure = (figure+1)/2*255
    figure = np.round(figure,0).astype(int)
    misc.imsave(path,figure)


img_gen = img_generator(imgs=imgs,batch_size=64)
img_z_samples = e_model.predict_generator(img_gen.__iter__(),steps=len(img_gen),verbose=True)

img_z_samples_ = img_z_samples/(img_z_samples**2).sum(1,keepdims=True)**0.5

# 相似采样函数
def sample_sim(path, n=8, k=2):
    figure = np.zeros((img_dim * (k+1), img_dim * n, 3))
    for i in range(n):
        idx = np.random.choice(len(imgs))
        # idxs = np.dot(img_z_samples_, img_z_samples_[idx]).argsort()[-k-1:][::-1]
        idxs = ((img_z_samples**2).sum(1) + (img_z_samples[idx]**2).sum() - 2 * np.dot(img_z_samples, img_z_samples[idx])).argsort()[:k+1]
        for j in range(k+1):
            digit = imread(imgs[idxs[j]])
            figure[j * img_dim:(j + 1) * img_dim,
                   i * img_dim:(i + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype(int)
    misc.imsave(path, figure)


if __name__ == '__main__':
    iters_per_sample = 100
    total_iter = 1000000
    n_size = 8
    img_data = img_generator(imgs, 'gan', batch_size).__iter__()
    Z = np.random.randn(n_size**2, z_dim)

    for i in range(total_iter):
        for j in range(2):
            x_sample = next(img_data)
            z_sample = np.random.randn(len(x_sample), z_dim)
            d_loss = d_train_model.train_on_batch(
                [x_sample, z_sample], None)
        for j in range(1):
            x_sample = img_data.next()
            z_sample = np.random.randn(len(x_sample), z_dim)
            g_loss = g_train_model.train_on_batch(
                [x_sample, z_sample], None)
        if i % 10 == 0:
            print('iter: %s, d_loss: %s, g_loss: %s' % (i, d_loss, g_loss))
        if i % iters_per_sample == 0:
            sample('samples/test_%s.png' % i, n_size, Z)
            sample_ae('samples/test_ae_%s.png' % i)
            g_train_model.save_weights('./g_train_model.weights')