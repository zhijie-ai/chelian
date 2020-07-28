#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/5/13 11:29                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
from GAN.SSGAN.impl3_tf.vlib.load_data import *
import tensorflow as tf
import argparse
import GAN.SSGAN.impl3_tf.train as train
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='DCGAN', help='DCGAN or WGAN-GP')
parser.add_argument('--trainable', type=bool, default=True,help='True for train and False for test')
parser.add_argument('--load_model', type=bool, default=False, help='True for load ckpt model and False for otherwise')
parser.add_argument('--label_num', type=int, default=2, help='the num of labled images we use， 2*100=200，batchsize:100')
args = parser.parse_args()

def main():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    sess = tf.InteractiveSession(config=config)
    model = train.Train(sess, args)
    if args.trainable:
        model.train()
    else:
        print( model.test())

if __name__ == '__main__':
    main()