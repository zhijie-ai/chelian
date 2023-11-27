#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Tencent Inc.
# Author: ranchodzu (ranchodzu@tencent.com)
# TIME: 2023/1/9 15:59
from tensorflow.keras.layers import *
import glob
import numpy as np
import imageio
from skimage.transform import resize



imgs = glob.glob('/apdcephfs_cq2/share_919031/ranchodzu/GAN/images/img_align_celeba/*.jpg')
imgs = glob.glob('../../images/img_align_celeba/*.jpg')
print(imgs)
np.random.shuffle(imgs)
img_dim=64

def imread(f):
    x = imageio.imread(f)
    x = resize(x, (img_dim, img_dim))
    return x.astype(np.float32)/ 255*2 - 1

def data_generator(batch_size=1024):
    X = []
    while True:
        np.random.shuffle(imgs)
        for f in imgs:
            X.append(imread(f))
            if len(X) == batch_size:
                X = np.array(X)
                yield X
                X = []


batch_size=64
img_generator = data_generator(batch_size)
print(next(img_generator))


z = Dense(512)
print(z.get_config())
