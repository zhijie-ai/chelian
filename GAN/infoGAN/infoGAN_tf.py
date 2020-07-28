#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/4/22 10:39                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

# 代码参考:https://www.liangzl.com/get-article-detail-12436.html

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import tensorflow.contrib.slim as slim


mnist = input_data.read_data_sets('../MNIST_data/')

tf.reset_default_graph()

class_dim = 10 # 10 classes
con_dim = 2 # 隐含信息变量维度
rand_dim = 38 # 噪声
batch_size=10
n_input = 784

x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.int32,[None])

def generator(z):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0
    print(z.get_shape())#(10,50)
    with tf.variable_scope('generator',reuse=reuse):
        x=slim.fully_connected(z,1024)
        print(x.shape)#(10,1024)
        x=slim.batch_norm(x,activation_fn=tf.nn.relu)
        print(x.shape)# (10,1024)
        x=slim.fully_connected(x,7*7*128)
        print(x.shape) # (10,6272)
        x=slim.batch_norm(x,activation_fn=tf.nn.relu)
        print(x.shape)#(10,6272)
        x=tf.reshape(x,[-1,7,7,128])
        print(x.shape)#(10,7,7,128)
        x=slim.conv2d_transpose(x,64,kernel_size=[4,4],stride=2,activation_fn=None)
        print(x.shape) #(10,14,14,64)
        x=slim.batch_norm(x,activation_fn=tf.nn.relu)
        print(x.shape) #(10,14,14,64)
        z = slim.conv2d_transpose(x,1,kernel_size=[4,4],stride=2,activation_fn=tf.nn.sigmoid)
        print('z--',z.shape) # (10,28,28,1)
        tf.nn.conv2d_transpose()
    return  z

def discriminator(x,num_classes=10,num_cont=2):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0
    print(x.get_shape()) #(10,28,28,1)
    with tf.variable_scope('discriminator' ,reuse=reuse):
        x = tf.reshape(x,shape=[-1,28,28,1])
        print(x.shape) #(10,28,28,1)
        x=slim.conv2d(x,num_outputs=64,kernel_size=[4,4],stride=2,activation_fn=tf.nn.leaky_relu)
        print(x.shape)##(10,14,14,64)
        x=slim.conv2d(x,num_outputs=128,kernel_size=[4,4],stride=2,activation_fn=tf.nn.leaky_relu)
        print(x.shape)#(10,7,7,128)
        x=slim.flatten(x)
        print(x.shape) # (10,6272)
        shared_tensor = slim.fully_connected(x,num_outputs=1024,activation_fn=tf.nn.leaky_relu)
        print(shared_tensor.shape) # (10,1024)
        recog_shared = slim.fully_connected(shared_tensor,num_outputs=128,activation_fn=tf.nn.leaky_relu)
        print(recog_shared.shape) #(10,128)

        disc = slim.fully_connected(shared_tensor,num_outputs=1,activation_fn=None)
        print(disc.shape) #(10,1)

        disc = tf.squeeze(disc,-1)
        print(disc.shape) #(10,)
        recog_cat = slim.fully_connected(recog_shared,num_outputs=num_classes,activation_fn=None)
        print(recog_cat.shape)#(10,10)

        recog_cont = slim.fully_connected(recog_shared,num_outputs=num_cont,activation_fn=tf.nn.sigmoid)
        print(recog_cont.shape) #(10,2)

    return disc,recog_cat,recog_cont


z_con = tf.random_normal((batch_size,con_dim))
print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
z_rand = tf.random_normal((batch_size,rand_dim))
z = tf.concat(values=[tf.one_hot(y,depth=class_dim),z_con,z_rand],axis=1)
gen = generator(z)
genout= tf.squeeze(gen,-1)

# labels for discriminator
y_real = tf.ones(batch_size)  # 真
y_fake = tf.zeros(batch_size)  # 假

#判别器
disc_real ,class_real,_ = discriminator(x)
disc_fake,class_fake,con_fake = discriminator(gen)
pred_class = tf.argmax(class_fake,dimension=1)

#判别器loss
loss_d_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,labels=y_real))
loss_d_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,labels=y_fake))
loss_d = (loss_d_r+loss_d_f)/2

#generator losss
loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,labels=y_real))
# categorical factor loss
loss_cf = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=class_fake,labels=y))# class ok 图片对不上
loss_cr = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=class_real,labels=y))# 生成的图片与class ok 与输入的class对不上
loss_c = (loss_cf+loss_cr)/2
#continuous factor loss
loss_con = tf.reduce_mean(tf.square(con_fake-z_con))

# 获得各个网络中各自的训练参数
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'discriminator' in var.name]
g_vars = [var for var in t_vars if 'generator' in var.name]

disc_global_step = tf.Variable(0,trainable=False)
gen_global_step = tf.Variable(0,trainable=False)

train_disc = tf.train.AdamOptimizer(0.0001).minimize(loss_d+loss_c +loss_con,var_list=d_vars,
                                                     global_step=disc_global_step)
train_gen = tf.train.AdamOptimizer(0.001).minimize(loss_g+loss_c+ loss_con,var_list=g_vars,
                                                   global_step=gen_global_step)

training_epochs = 3
display_step = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        #遍历全部数据集
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            print('AAAAAAAAA',batch_ys.shape)
            feeds = {x:batch_xs,y:batch_ys}

            # Fit training using batch data
            l_disc,_,l_d_step = sess.run([loss_d,train_disc,disc_global_step],feeds)
            l_gen,_,l_g_step = sess.run([loss_g,train_gen,gen_global_step],feeds)

        # 显示训练中的详细信息
        if epoch %display_step == 0:
            print('Epoch:','%04d' %(epoch+1),"cost=","{:.9f}".format(l_disc),l_gen)

    print('完成.........')

    #测试
    print('Result:',loss_d.eval({x:mnist.test.images[:batch_size],y:mnist.test.labes[:batch_size]}),
          loss_g.eval({x:mnist.test.images[:batch_size],y:mnist.test.labels[:batch_size]}))

    #根据图片模拟生成图片
    show_num = 10
    gensimple,d_class,inputx,inputy,con_out = sess.run([genout,pred_class,x,y,con_fake],
                                                       feed_dict={x:mnist.test.images[:batch_size],
                                                                  y:mnist.test.labes[:batch_size]})
    f,a = plt.subplots(2,10,figsize=(10,2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(inputx[i],(28,28)))
        a[1][i].imshow(np.reshape(gensimple[i],(28,28)))
        print('d_class',d_class[i],'inputy',inputy[i],'con_out',con_out[i])

    plt.draw()
    plt.show()


    my_con = tf.placeholder(tf.float32,[batch_size,2])
    myz = tf.concat(axis=1,values=[tf.one_hot(y,depth=class_dim),my_con,z_rand])
    mygen=generator(myz)
    mygenout = tf.squeeze(mygen,-1)

    my_con1 = np.ones([10,2])
    a = np.linspace(0.0001,0.9999,10)
    y_input = np.ones([10])
    figure = np.zeros((28*10,28*10))
    my_rand = tf.random_normal((10,rand_dim))
    for i in range(10):
        for j in range(10):
            my_con1[j][0]=a[i]
            my_con1[j][1]=a[j]
            y_input[j] = j
        mygenoutv = sess.run(mygenout,feed_dict={y:y_input,my_con:my_con1})
        for jj in range(10):
            digit = mygenoutv[jj].reshape(28,28)
            figure[i*28:(i+1)*28,jj*28:(jj+1)*28] = digit

    plt.figure(figsize=(10,10))
    plt.imshow(figure,cmap='Greys_r')
    plt.show()








