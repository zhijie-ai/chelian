# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/3/18 9:17                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

slim=tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli
OneHotCategorical = tf.contrib.distributions.OneHotCategorical
RelaxedOneHotCategorical = tf.contrib.distributions.RelaxedOneHotCategorical

# black-on-white MNIST (harder to learn than white-on-black MNIST)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size=100
tau0=1.0 # initial temperature
K=10 # number of classes
N=200//K # number of categorical distributions
straight_through=False # if True, use Straight-through Gumbel-Softmax
kl_type='relaxed' # choose between ('relaxed', 'categorical')
learn_temp=False

x=tf.placeholder(tf.float32, shape=(batch_size,784), name='x')
net = tf.cast(tf.random_uniform(tf.shape(x)) < x, x.dtype) # dynamic binarization
net = slim.stack(net,slim.fully_connected,[512,256])
logits_y = tf.reshape(slim.fully_connected(net,K*N,activation_fn=None),[-1,N,K])
tau = tf.Variable(tau0,name="temperature",trainable=learn_temp)
q_y = RelaxedOneHotCategorical(tau,logits_y)
y = q_y.sample()
if straight_through:
    y_hard = tf.cast(tf.one_hot(tf.argmax(y,-1),K), y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
net = slim.flatten(y)
net = slim.stack(net,slim.fully_connected,[256,512])
logits_x = slim.fully_connected(net,784,activation_fn=None)
p_x = Bernoulli(logits=logits_x)
x_mean = p_x.mean()

recons = tf.reduce_sum(p_x.log_prob(x),1)
logits_py = tf.ones_like(logits_y) * 1./K

if kl_type=='categorical' or straight_through:
    # Analytical KL with Categorical prior
    p_cat_y = OneHotCategorical(logits=logits_py)
    q_cat_y = OneHotCategorical(logits=logits_y)
    KL_qp = tf.contrib.distributions.kl(q_cat_y, p_cat_y)
else:
    # Monte Carlo KL with Relaxed prior
    p_y = RelaxedOneHotCategorical(tau,logits=logits_py)
    KL_qp = q_y.log_prob(y) - p_y.log_prob(y)

KL = tf.reduce_sum(KL_qp,1)
mean_recons = tf.reduce_mean(recons)
mean_KL = tf.reduce_mean(KL)
loss = -tf.reduce_mean(recons-KL)

train_op=tf.train.AdamOptimizer(learning_rate=3e-4).minimize(loss)

data = []
with tf.train.MonitoredSession() as sess:
    for i in range(1,50000):
        batch = mnist.train.next_batch(batch_size)
        res = sess.run([train_op, loss, tau, mean_recons, mean_KL], {x : batch[0]})
        if i % 100 == 1:
            data.append([i] + res[1:])
        if i % 1000 == 1:
            print('Step %d, Loss: %0.3f' % (i,res[1]))
    # end training - do an eval
    batch = mnist.test.next_batch(batch_size)
    np_x = sess.run(x_mean, {x : batch[0]})

data = np.array(data).T

f,axarr=plt.subplots(1,4,figsize=(18,6))
axarr[0].plot(data[0],data[1])
axarr[0].set_title('Loss')

axarr[1].plot(data[0],data[2])
axarr[1].set_title('Temperature')

axarr[2].plot(data[0],data[3])
axarr[2].set_title('Recons')

axarr[3].plot(data[0],data[4])
axarr[3].set_title('KL')

tmp = np.reshape(np_x,(-1,280,28)) # (10,280,28)
img = np.hstack([tmp[i] for i in range(10)])
plt.imshow(img)
plt.grid('off')