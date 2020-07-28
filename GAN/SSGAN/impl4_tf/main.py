#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/6/24 21:27                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc as scm
from SSGAN.impl3_tf.vlib.layers import *
import tensorflow as tf
import numpy as np
import os
from SSGAN.impl4_tf import model

trainable = True

def main():
    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    sess = tf.InteractiveSession(config=config)
    ssgan = model.Model(sess)
    if trainable:
        ssgan.train()
    else:
        print (ssgan.test())

if __name__ == '__main__':
    main()