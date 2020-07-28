#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/4/2 17:25                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

import numpy as np

from numpy_utils import make_one_hot

def create_continuous_noise(num_conditions,style_size,size):
    continuous = np.random.uniform(-1.0,1,0,size=(size,num_conditions))
    style = np.random.standard_normal(size=(size,style_size))
    return np.hstack([continuous,style])

def create_categorical_noise(categorical_cardinality,size):
    noise=[]
    for cardinality in categorical_cardinality:
        noise.append(
            np.random.randint(0,cardinality,size=size)
        )
    return noise

def encode_infogan_noise(categorical_cardinality,categorical_samples,continuous_samples):
    noise=[]
    for cardinality, sample in zip(categorical_cardinality,categorical_samples):
        noise.append(make_one_hot(sample,size=cardinality))
    noise.append(continuous_samples)
    return np.hstack(noise)

def create_infogan_noise_sample(categorical_cardinality,num_continuous,style_size):
    def sample(batch_size):
        return encode_infogan_noise(
            categorical_cardinality,
            create_categorical_noise(categorical_cardinality,size=batch_size),
            create_continuous_noise(num_continuous,style_size,size=batch_size)
        )

    return sample

def create_gan_noise_sample(style_size):
    def sample(batch_size):
        return np.random.standard_normal(size=(batch_size,style_size))
    return sample
