#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/4/25 9:51                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim


def leaky_relu(x):
    print('============================',x.get_shape())
    return tf.where(tf.greater(x,0),0.1*x)

def discriminator(inputs,params):
    with slim.arg_scope([layers.fully_connected,layers.conv2d],
                        activation_fn = tf.nn.relu,normalizer_fn=None,
                        normalizer_params={'updates_collections':None,
                                           'is_training':True,'decay':0.9}):
        with tf.variable_scope('discriminator',reuse=tf.AUTO_REUSE):
            with tf.variable_scope('shared'):
                d1 = layers.conv2d(inputs,64,[4,4],stride=2)
                d2 = layers.conv2d(d1,128,[4,4],stride=2,normalizer_fn=layers.batch_norm)

                d2_flatten = layers.flatten(d2)

                d3 = layers.fully_connected(d2_flatten,1024,normalizer_fn=layers.batch_norm)
            with tf.variable_scope('d'):
                d_out = layers.fully_connected(d3,1,activation_fn=None)
                d_out = tf.squeeze(d_out,-1)

            with tf.variable_scope('q'):
                r1 = layers.fully_connected(d3,128,normalizer_fn=layers.batch_norm)
                r_cat = layers.fully_connected(r1,params.num_category,activation_fn=None)
                r_cont_mu = layers.fully_connected(r1,params.num_cont,activation_fn=None)
                if params.fix_var:
                    r_cont_var = 1
                else:
                    r_cont_logvar = layers.fully_connected(r1,params.num_cont,activation_fn=None)
                    r_cont_var = tf.exp(r_cont_logvar)

            return d_out,r_cat,r_cont_mu,r_cont_var