#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/4/30 14:07                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np
import tensorflow as tf

from SSGAN.impl1_tf.util import log
from .ops import conv2d
from .ops import fc

class Discriminator():
    def __init__(self,name,num_class,norm_type,is_train):
        self.name = name
        self._num_class = num_class
        self._norm_type = norm_type
        self._is_train = is_train
        self._resue = False

    def __call__(self, input):
        with tf.variable_scope(self.name,reuse=self._resue):
            if not self._resue:
                print('\033[93m'+self.name+'\033[0m')
            _ = input
            print('RRRRRRRRRRRRRR_input',input.shape)
            num_channel = [32,64,128,256,256,512]
            num_layer = np.ceil(np.log2(min(_.shape.as_list()[1:3]))).astype(np.int)

            for i in range(num_layer):
                ch = num_channel[i] if i < len(num_channel) else 512
                _ = conv2d(_,ch,self._is_train,info=not self._resue,
                           norm = self._norm_type,name='conv{}'.format(i+1))
            _ = conv2d(_,int(num_channel[i]/4),self._is_train,k=1,s=1,info= not self._resue,
                       activation_fn=None,norm='None',name='conv{}'.format(i+3))
            print('AAAAAAAAAAAAAAAAAA', _.shape.as_list())
            _ = tf.squeeze(_)
            print('AAAAAAAAAAAAAAAAAA',_.shape.as_list(),i)
            _ = fc(_,self._num_class+1,self._is_train,info=not self._resue,
                           norm = self._norm_type,name='fc{}'.format(i+1))
            print('AAAAAAAAAAAAAAAAAAB', _.shape.as_list(), i)
            if not self._resue:
                log.info('discriminator output {}'.format(_.shape.as_list()))
            self._resue = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,self.name)
            print('WWWWWWWWWWWWWWWW',_.shape)
            return tf.nn.sigmoid(_),_