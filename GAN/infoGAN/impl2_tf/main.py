#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/4/25 17:33                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

import tensorflow as tf
from GAN.infoGAN.impl2_tf.model import InfoGan

flags=tf.app.flags

flags.DEFINE_string('model_dir','model_dir','model directory')
flags.DEFINE_boolean('is_training',True,'whether it is training or inferecing')
flags.DEFINE_boolean('fix_var',True,'whether to approximate variance')
flags.DEFINE_integer('num_category',10,'category dim of latent variable')
flags.DEFINE_integer('num_cont',2,'continuous dim of latent variable')
flags.DEFINE_integer('batch_size',32,'batch_size')
flags.DEFINE_integer('epoch',50,'epochs')
flags.DEFINE_integer('num_rand',62,'random noise dim of latent variable')
flags.DEFINE_float('d_lr',2e-4,'learning rate for discriminator')
flags.DEFINE_float('g_lr',1e-3,'learning rate for generator')

FLAGS = flags.FLAGS

def main(unused_argv):
    config = tf.ConfigProto()
    # config.gpu_option.allow_growth = True
    sess = tf.Session(config=config)

    model=InfoGan(sess,FLAGS)

    model.train() if FLAGS.is_training else model.inference()

if __name__ == '__main__':
    # tf.app.run()
    main(None)