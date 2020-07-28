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

# D的输出是原本的num_class+1，如果是监督学习，只取前num_class的输出，用交叉熵损失函数来学习，如果是无监督
# 学习，则用普通的GAN的损失
import pickle
from SSGAN.impl3_tf.vlib.layers import *
import tensorflow as tf
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class Model(object):
    def __init__(self, sess):

        self.sess = sess
        self.img_size = 100
        self.trainable = True
        self.batch_size = 32
        self.lr = 0.01
        self.mm = 0.5
        self.z_dim = 100
        self.EPOCH = 100
        self.dim = 1
        self.num_class = 3
        self.load_model = False
        self.build_model()

    def build_model(self):
        # build  placeholders
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_size, self.img_size, self.dim],
                                name='real_img')
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim, self.z_dim, self.dim], name='noise')
        self.label = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_class - 1], name='label')
        self.flag = tf.placeholder(tf.float32, shape=[], name='flag')

        # define the network
        self.G_img = self.generator('gen', self.z, reuse=False)
        d_logits_r, layer_out_r = self.discriminator('dis', self.x, reuse=False)
        d_logits_f, layer_out_f = self.discriminator('dis', self.G_img, reuse=True)

        f_match = tf.constant(0., dtype=tf.float32)
        for i in range(4):
            f_match += tf.reduce_mean(tf.multiply(layer_out_f[i] - layer_out_r[i], layer_out_f[i] - layer_out_r[i]))

        #self.label 如果flag是1的话，就是监督学习，如果是0的话，就是无监督学习
        self.d_loss = 10 * self.flag * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=d_logits_r[:, :-1])) \
                      + (1 - self.flag) * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_r[:, -1], labels=tf.ones_like(d_logits_r[:, -1]))) \
                      + (1 - self.flag) * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_f[:, -1], labels=tf.zeros_like(d_logits_f[:, -1])))

        self.g_loss = (1 - self.flag) * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_f[:, -1], labels=tf.ones_like(d_logits_r[:, -1])))

        all_vars = tf.global_variables()
        g_vars = [v for v in all_vars if 'gen' in v.name]
        d_vars = [v for v in all_vars if 'dis' in v.name]

        self.opt_d = tf.train.AdamOptimizer(self.lr, beta1=self.mm).minimize(self.d_loss, var_list=d_vars)
        self.opt_g = tf.train.AdamOptimizer(self.lr, beta1=self.mm).minimize(self.g_loss, var_list=g_vars)
        # test
        test_logits, _ = self.discriminator('dis', self.x, reuse=True)
        test_logits = tf.nn.softmax(test_logits[:, :-1])

        self.prediction = tf.nn.in_top_k(test_logits, tf.argmax(self.label, axis=1), 1)

        self.saver = tf.train.Saver()
        if not self.load_model:
            init = tf.global_variables_initializer()
            self.sess.run(init)
        elif self.load_model:
            self.saver.restore(self.sess, os.getcwd() + '/model_saved/model.ckpt')
            print('model load done')
        self.sess.graph.finalize()

    def train(self):
        if not os.path.exists('model_saved'):
            os.mkdir('model_saved')
        if not os.path.exists('gen_picture'):
            os.mkdir('gen_picture')

        temp = 0.50
        print('training')

        f = open('training_data.pkl', 'rb')
        images1 = pickle.load(f)
        f.close()
        images1 = np.array(images1)

        f = open('point_data.pkl', 'rb')
        images2 = pickle.load(f)
        f.close()
        images2 = images2 / 50

        lab = []
        for line in open('matrix_flagpic.txt'):
            seg = line.strip("\n").split('\t')
            lab = lab + [int(seg[1])] * 8
        lab = np.array(lab)

        X_train, self.X_test, y_train, self.y_test = train_test_split(images2, lab, test_size=0.2, random_state=2018)

        y_train = np.expand_dims(y_train, 1)
        self.y_test = np.expand_dims(self.y_test, 1)

        enc = OneHotEncoder(dtype=np.float32, sparse=False)
        y_train = enc.fit_transform(y_train)
        self.y_test = enc.fit_transform(self.y_test)

        print(y_train.shape, self.y_test.shape)

        X_train = np.concatenate((X_train, images1))
        X_train = X_train * 2 - 1

        self.X_test = self.X_test * 2 - 1

        del images1, images2, lab

        X_train = np.expand_dims(X_train, 3)
        self.X_test = np.expand_dims(self.X_test, 3)

        print(X_train.shape, self.X_test.shape)

        for epoch in range(self.EPOCH):

            iters = int(6543 // self.batch_size)
            for idx in range(iters):
                start_t = time.time()
                flag = 1 if idx < 20 else 0

                batchx = X_train[idx * self.batch_size: (idx + 1) * self.batch_size]
                noise = np.random.normal(-1, 1, [self.batch_size, 100, 100, 1])

                if flag == 1:
                    batchl = y_train[idx * self.batch_size: (idx + 1) * self.batch_size]
                else:
                    batchl = np.ones(shape=(self.batch_size, 2))

                g_opt = [self.opt_g, self.g_loss]
                d_opt = [self.opt_d, self.d_loss]
                feed = {self.x: batchx, self.z: noise, self.label: batchl, self.flag: flag}
                # update the Discrimater k times
                _, loss_d = self.sess.run(d_opt, feed_dict=feed)
                # update the Generator one time

                _, loss_g = self.sess.run(g_opt, feed_dict=feed)
                # print ("[%3f][epoch:%2d/%2d][iter:%4d/%4d],loss_d:%5f,loss_g:%4f, d1:%4f, d2:%4f"%
                #       (time.time()-start_t, epoch, self.EPOCH,idx,iters, loss_d, loss_g,d1,d2)), 'flag:',flag

                if (idx + 1) % 50 == 0:
                    print('images saving............')
                    img = self.sess.run(self.G_img, feed_dict=feed)
                    img = (img + 1) / 2

                    output = open(os.getcwd() + '/gen_picture/' + 'sample{}_{}.pkl' \
                                  .format(epoch, (idx + 1) / 50), 'wb')
                    pickle.dump(img, output)
                    print('images save done')
            test_acc = self.test()

            print('test acc:{}'.format(test_acc), 'temp:%3f' % (temp))
            if test_acc > temp:
                print('model saving..............')
                path = os.getcwd() + '/model_saved'
                save_path = os.path.join(path, "model.ckpt")
                self.saver.save(self.sess, save_path=save_path)
                print('model saved...............')
                temp = test_acc

    def generator(self, name, noise, reuse):
        with tf.variable_scope(name, reuse=reuse):
            l = self.batch_size
            s = self.img_size
            s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8) + 1, int(s / 16) + 1

            e0 = conv2d('g_e_con0', noise, 5, 32, stride=1, padding='SAME')
            e0 = self.bn('g_e_bn0', e0)

            e1 = conv2d('g_e_con1', tf.nn.relu(e0), 5, 64, stride=2, padding='SAME')
            e1 = self.bn('g_e_bn1', e1)

            e2 = conv2d('g_e_con2', tf.nn.relu(e1), 5, 64 * 2, stride=2, padding='SAME')
            e2 = self.bn('g_e_bn2', e2)

            e3 = conv2d('g_e_con3', tf.nn.relu(e2), 5, 64 * 3, stride=2, padding='SAME')
            e3 = self.bn('g_e_bn3', e3)

            e4 = conv2d('g_e_con4', tf.nn.relu(e3), 5, 64 * 4, stride=2, padding='SAME')
            e4 = self.bn('g_e_bn4', e4)

            e5 = conv2d('g_e_con5', tf.nn.relu(e4), 5, 64 * 4, stride=2, padding='SAME')
            e5 = self.bn('g_e_bn5', e5)

            d1 = deconv2d('g_d_dcon1', tf.nn.relu(e5), 5, outshape=[l, s16, s16, 64 * 4])
            d1 = self.bn('g_d_bn1', d1)
            # d1 = tf.nn.dropout(d1, 0.5)
            d1 = tf.concat([d1, e4], 3)

            d2 = deconv2d('g_d_dcon2', tf.nn.relu(d1), 5, outshape=[l, s8, s8, 64 * 3])
            d2 = self.bn('g_d_bn2', d2)
            # d2 = tf.nn.dropout(d2, 0.5)
            d2 = tf.concat([d2, e3], 3)

            d3 = deconv2d('g_d_dcon3', tf.nn.relu(d2), 5, outshape=[l, s4, s4, 64 * 2])
            d3 = self.bn('g_d_bn3', d3)
            # d3 = tf.nn.dropout(d3, 0.5)
            d3 = tf.concat([d3, e2], 3)

            d4 = deconv2d('g_d_dcon4', tf.nn.relu(d3), 5, outshape=[l, s2, s2, 64 * 1])
            d4 = self.bn('g_d_bn4', d4)
            # d4 = tf.nn.dropout(d4, 0.5)
            d4 = tf.concat([d4, e1], 3)

            d5 = deconv2d('g_d_dcon5', tf.nn.relu(d4), 5, outshape=[l, s, s, 32])
            d5 = self.bn('g_d_bn5', d5)
            d5 = tf.concat([d5, e0], 3)

            ret = conv2d('g_ret_con', tf.nn.relu(d5), 5, self.dim, stride=1, padding='SAME')
            # ret = self.bn('g_ret_bn', ret)

            return tf.nn.tanh(ret)

    def discriminator(self, name, inputs, reuse):
        with tf.variable_scope(name, reuse=reuse):
            out = []

            output = inputs
            output = relu('d_pre_relu1',
                          self.bn('d_pre_bn1', conv2d('d_pre_con1', output, 5, 64, stride=1, padding='SAME')))
            output1 = conv2d('d_con1', output, 5, 64, stride=1, padding='SAME')
            output1 = relu('d_relu1', self.bn('d_bn1', output1))
            output1 = conv2d('d_con2', output1, 5, 64, stride=1, padding='SAME')
            output1 = relu('d_relu2', self.bn('d_bn2', output1))
            output1 = output1 + output
            output1 = relu('d_relu2_1', self.bn('d_bn2_1', output1))
            output1 = max_pool(output1)
            out.append(output1)

            output = output1
            output = relu('d_pre_relu2',
                          self.bn('d_pre_bn2', conv2d('d_pre_con2', output, 3, 64 * 2, stride=1, padding='SAME')))
            output2 = conv2d('d_con3', output, 3, 64 * 2, stride=1, padding='SAME')
            output2 = relu('d_relu3', self.bn('d_bn3', output2))
            output2 = conv2d('d_con4', output2, 3, 64 * 2, stride=1, padding='SAME')
            output2 = relu('d_relu4', self.bn('d_bn4', output2))
            output2 = output2 + output
            output2 = relu('d_relu4_1', self.bn('d_bn4_1', output2))
            output2 = max_pool(output2)
            out.append(output2)

            output = output2
            output = relu('d_pre_relu3',
                          self.bn('d_pre_bn3', conv2d('d_pre_con3', output, 3, 64 * 3, stride=1, padding='SAME')))
            output3 = conv2d('d_con5', output, 3, 64 * 3, stride=1, padding='SAME')
            output3 = relu('d_relu5', self.bn('d_bn5', output3))
            output3 = conv2d('d_con6', output3, 3, 64 * 3, stride=1, padding='SAME')
            output3 = relu('d_relu6', self.bn('d_bn6', output3))
            output3 = output3 + output
            output3 = relu('d_relu6_1', self.bn('d_bn6_1', output3))
            output3 = max_pool(output3)
            out.append(output3)

            output = output3
            output = relu('d_pre_relu4',
                          self.bn('d_pre_bn4', conv2d('d_pre_con4', output, 3, 64 * 4, stride=1, padding='SAME')))
            output4 = conv2d('d_con7', output, 3, 64 * 4, stride=1, padding='SAME')
            output4 = relu('d_relu7', self.bn('d_bn7', output4))
            output4 = conv2d('d_con8', output4, 3, 64 * 4, stride=1, padding='SAME')
            output4 = relu('d_relu8', self.bn('d_bn8', output4))
            output4 = output4 + output
            output4 = relu('d_relu8_1', self.bn('d_bn8_1', output4))
            output4 = max_pool(output4)
            out.append(output4)

            output = output4
            output5 = conv2d('d_con9', output, 3, 64 * 4, stride=1, padding='SAME')
            output5 = relu('d_relu9', self.bn('d_bn9', output5))
            output5 = conv2d('d_con10', output5, 3, 64 * 4, stride=1, padding='SAME')
            output5 = relu('d_relu10', self.bn('d_bn10', output5))
            output5 = output5 + output
            output5 = relu('d_relu10_1', self.bn('d_bn10_1', output5))
            output5 = max_pool(output5)
            out.append(output5)

            output = output5
            output6 = conv2d('d_con11', output, 3, 64 * 4, stride=1, padding='SAME')
            output6 = relu('d_relu11', self.bn('d_bn11', output6))
            output6 = conv2d('d_con12', output6, 3, 64 * 4, stride=1, padding='SAME')
            output6 = relu('d_relu12', self.bn('d_bn12', output6))
            output6 = output6 + output
            output6 = relu('d_relu12_1', self.bn('d_bn12_1', output6))
            output6 = conv2d('d_con12_1', output6, 3, 64 * 4, stride=2, padding='SAME')
            output6 = relu('d_relu12_2', self.bn('d_bn12_2', output6))
            out.append(output6)

            output = tf.reshape(output6, [self.batch_size, 2 * 2 * 64 * 4])
            output = fc('d_fc', output, self.num_class)

            return output, out

    def bn(self, name, input):
        val = tf.contrib.layers.batch_norm(input, decay=0.9,
                                           updates_collections=None,
                                           epsilon=1e-5,
                                           scale=True,
                                           is_training=True,
                                           scope=name)
        return val

    def test(self):
        count = 0.
        print('testing...')
        for i in range(160 // self.batch_size):
            testx = self.X_test[i * self.batch_size: (i + 1) * self.batch_size]
            testl = self.y_test[i * self.batch_size: (i + 1) * self.batch_size]
            prediction = self.sess.run(self.prediction, feed_dict={self.x: testx, self.label: testl})
            count += np.sum(prediction)


        return count / 160.