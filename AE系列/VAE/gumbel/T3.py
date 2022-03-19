# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/3/19 16:20                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import tensorflow as tf
sess = tf.Session()

def differentiable_sample(logits, temperature=1):
    noise = tf.random_uniform(tf.shape(logits), seed=11)
    logits_with_noise = logits - tf.log(-tf.log(noise))
    return tf.nn.softmax(logits_with_noise / temperature)

mean = tf.Variable(2.)
mean2 = tf.Variable(2.)
idxs = tf.Variable([0., 1., 2., 3., 4.])
# An unnormalised approximately-normal distribution
logits = tf.exp(-(idxs - mean) ** 2)
sess.run(tf.global_variables_initializer())

def print_logit_vals():
    logit_vals = sess.run(logits)
    print(" ".join(["{:.2f}"] * len(logit_vals)).format(*logit_vals))

print("Logits: ")
print_logit_vals()

sample = differentiable_sample(logits)
sample_weights = tf.Variable([1., 2., 3., 4., 5.], trainable=False)
result = tf.reduce_sum(sample * sample_weights)

sess.run(tf.global_variables_initializer())
train_op = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(-result, var_list=[mean2,mean])

print("Distribution mean: {:.2f}".format(sess.run(mean)))
for i in range(5):
    sess.run(train_op)
    print("Distribution mean: {:.2f}".format(sess.run(mean)))