# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/3/18 14:19                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(42)
tf.set_random_seed(42)

BATCHS_IN_EPOCH = 100
BATCH_SIZE = 10
EPOCHS = 200  # the stream is infinite so one epoch will be defined as BATCHS_IN_EPOCH * BATCH_SIZE
GENERATOR_TRAINING_FACTOR = 10  # for every training of the disctiminator we'll train the generator 10 times
LEARNING_RATE = 0.0007
TEMPERATURE = 0.01  # we use a constant, but for harder problems we should anneal it

number_to_prob = {
    0: 0.0,
    1: 0.0,
    2: 0.1,
    3: 0.3,
    4: 0.6
}

def generate_text():
    while True:
        yield np.random.choice(list(number_to_prob.keys()), p=list(number_to_prob.values()), size=1)


dataset = tf.data.Dataset.from_generator(generate_text,
                                         output_types=tf.int32,
                                         output_shapes=1).batch(BATCH_SIZE)
value = dataset.make_one_shot_iterator().get_next()
value = tf.one_hot(value, len(number_to_prob))
value = tf.squeeze(value, axis=1)

def generator():
    with tf.variable_scope('generator'):
        logits = tf.get_variable('logits', initializer=tf.ones([len(number_to_prob)]))
        gumbel_dist = tf.contrib.distributions.RelaxedOneHotCategorical(TEMPERATURE, logits=logits)
        probs = tf.nn.softmax(logits)
        generated = gumbel_dist.sample(BATCH_SIZE)
        return generated, probs

def discriminator(x):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        return tf.contrib.layers.fully_connected(x,
                                                 num_outputs=1,
                                                 activation_fn=None)

generated_outputs, generated_probs = generator()
discriminator1 = discriminator(value)
discriminated_real = discriminator1
discriminated_generated = discriminator(generated_outputs)

d_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminated_real,
                                            labels=tf.ones_like(discriminated_real)))
d_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminated_generated,
                                            labels=tf.zeros_like(discriminated_generated)))
d_loss = d_loss_real + d_loss_fake

g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminated_generated,
                                            labels=tf.ones_like(discriminated_generated)))

all_vars = tf.trainable_variables()
g_vars = [var for var in all_vars if var.name.startswith('generator')]
d_vars = [var for var in all_vars if var.name.startswith('discriminator')]

d_train_opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(g_loss, var_list=g_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(generated_outputs))
    learned_probs = []
    for _ in range(EPOCHS):
        for _ in range(BATCHS_IN_EPOCH):
            sess.run(d_train_opt)
        for _ in range(GENERATOR_TRAINING_FACTOR):
            sess.run(g_train_opt)
        learned_probs.append(sess.run(generated_probs))

print(np.array(learned_probs))
plt.figure(figsize=(10, 2))
prob_errors = [np.array(learned_prob) - np.array(list(number_to_prob.values()))
               for learned_prob in learned_probs]
plt.imshow(np.transpose(prob_errors),
           cmap='bwr',
           aspect='auto',
           vmin=-2,
           vmax=2)
plt.xlabel('epoch')
plt.ylabel('number')
plt.colorbar(aspect=10, ticks=[-2, 0, 2])
plt.show()