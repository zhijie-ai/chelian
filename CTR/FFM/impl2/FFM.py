#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/11/26 14:39                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import sys

from .tools import transfer_data, get_batch

class Args():
    # number of latent factors
    k = 6
    # number of fields
    f =24
    # num of features
    p = 100
    learning_rate = 0.1
    batch_size = 64
    l2_reg_rate = 0.001
    feature2field=None
    checkpoint_dir = 'logs/saver'
    is_training = True
    epoch = 1

class Model():
    def __init__(self,args):
        self.k = args.k
        self.f = args.f
        self.p = args.p
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.l2_reg_rate = args.l2_reg_rate
        self.feature2field = args.feature2field
        self.checkpoint_dir = args.checkpoint_dir

    def build_model(self):
        self.X = tf.placeholder('float32',[self.batch_size,self.p])
        self.y = tf.placeholder('float32',[None,1])

        # linear part
        with tf.variable_scope('linear_layer'):
            b = tf.get_variable('bias',shape=[1],
                                initializer=tf.zeros_initializer())
            self.w1 = tf.get_variable('w1',shape=[self.p,1],
                                      initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01))
            # shape of [None,1]
            self.linear_terms = tf.add(tf.matmul(self.X,self.w1),b)
            print('self.linear_terms:')
            print(self.linear_terms)

        with tf.variable_scope('nolinear_layer'):
            self.v = tf.get_variable('v',shape=[self.p,self.f,self.k],dtype='float32',
                                     initializer=tf.truncated_normal_initializer(0,0.01))
            # v:pxfxk
            self.field_cross_interaction = tf.constant(0,dtype='float32')
            # 每个特征
            for i in range(self.p):
                # 寻找没有match过的特征，也就是论文中的j=i+1开始
                for j in range(i+1,self.p):
                    # vifj
                    vifj = self.v[i,self.feature2field[j]]
                    # vjfi
                    vjfi = self.v[j,self.feature2field[i]]
                    # vi*vj
                    vivj = tf.reduce_sum(tf.multiply(vifj,vjfi))
                    # xi*xj
                    xixj = tf.multiply(self.X[:,i],self.X[:,j])
                    self.field_cross_interaction += tf.multiply(vivj,xixj)
            self.field_cross_interaction = tf.reshape(self.field_cross_interaction, (self.batch_size, 1))
            print('self.field_cross_interaction:')
            print(self.field_cross_interaction)

        self.y_out = tf.add(self.linear_terms,self.field_cross_interaction)
        print('y_out_prob:')
        print(self.y_out)
        # -1/1 情况下的logistic loss
        self.loss = tf.reduce_mean(tf.log(1+tf.exp(-self.y*self.y_out)))

        # 正则：sum(w^2)/2*l2_reg_rate
        # 这边只加了weight，有需要的可以加上bias部分
        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.w1)
        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.v)
        self.global_step  = tf.Variable(0,trainable=False)
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        trainable_params = tf.trainable_variables()
        print(trainable_params)
        gradients = tf.gradients(self.loss,trainable_params)
        clip_gradients,_ = tf.clip_by_global_norm(gradients,5)
        self.train_op = opt.apply_gradients(zip(clip_gradients,trainable_params),
                                            global_step=self.global_step)

    def train(self,sess,x,label):
        loss,_,step = self.run([self.loss,self.train_op,self.global_step],feed_dict={
            self.X:x,
            self.y:label
        })
        return loss,step

    def cal(self,sess,x,label):
        y_out_prob_ = sess.run([self.y_out],feed_dict={
            self.X:x,
            self.y:label
        })
        return y_out_prob_,label

    def predict(self,sess,x):
        result = sess.run([self.y_out],feed_dict={self.X:x})
        return result

    def save(self,sess,path):
        saver = tf.train.Saver()
        saver.restore(sess,save_path=path)

if __name__ == '__main__':
    # loading base params
    args = Args()
    # loading base data
    train_data_path = '/Users/slade/Documents/Code/machine-learning/data/avazu_CTR/train_sample.csv'
    train_data = pd.read_csv(train_data_path)
    train_data['click'] = train_data['click'].map(lambda x: -1 if x == 0 else x)
    # loading feature2field dict
    with open('sets/feature2field.pkl', 'rb') as f:
        args.feature2field = pickle.load(f)

    fields = ['C1', 'C18', 'C16', 'click']

    fields_dict={}
    for field in fields:
        with open('sets/' + field + '.pkl', 'rb') as f:
            fields_dict[field] = pickle.load(f)

    args.f = len(fields)-1
    print('f:%s'%args.f)
    args.p = max(fields_dict['click'].values()) - 1
    print('p:%s' % (max(fields_dict['click'].values()) - 1))
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    all_len = max(fields_dict['click'].values()) + 1

    cnt = train_data.shape[0]//args.batch_size
    with tf.Session(config=gpu_config) as sess:
        model = Model(args)
        model.build_model()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if args.is_training:
            # batch_size data
            for i in range(args.epoch):
                for j in range(cnt):
                    data = get_batch(train_data, args.batch_size, j)
                    actual_batch_size = len(data)
                    batch_X = []
                    batch_y = []
                    for k in range(actual_batch_size):
                        sample = data.iloc[k, :]
                        array = transfer_data(sample, fields_dict, all_len)
                        # 最后两位[0,-1]:label=0,[0,1]:label=1
                        batch_X.append(array[:-2])
                        # 最后一位即为label
                        batch_y.append(array[-1])

                    batch_X = np.array(batch_X)
                    batch_y = np.array(batch_y)
                    batch_y = batch_y.reshape(args.batch_size, 1)
                    loss,step = model.train(sess,batch_X,batch_y)
                    if j %100==0:
                        print('the times of training is %d, and the loss is %s' % (j, loss))
                        model.save(sess,args.checkpoint_dir)

        else:
            model.restore(sess, args.checkpoint_dir)
            for j in range(cnt):
                data = get_batch(train_data, args.batch_size, j)
                actual_batch_size = len(data)
                batch_X = []
                for k in range(actual_batch_size):
                    sample = data.iloc[k, :]
                    array = transfer_data(sample, fields_dict, all_len)
                    batch_X.append(array[:-2])
                batch_X = np.array(batch_X)
                result = model.predict(sess, batch_X)
                print(result)




