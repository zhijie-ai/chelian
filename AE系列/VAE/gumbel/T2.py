# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/3/19 16:13                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

def softmax(logits):
    max_value = np.max(logits)
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp)
    dist = exp / exp_sum
    return dist

def sample_with_softmax(logits, size):
    pros = softmax(logits)
    return np.random.choice(len(logits), size, p=pros)

def inv_gumbel_cdf(y, mu=0, beta=1, eps=1e-20):
    return mu - beta * np.log(-np.log(y + eps))

def gumbel_cdf(x, mu=0, beta=1):
    z = (x - mu) / beta
    return np.exp(-np.exp(-z))

print(inv_gumbel_cdf(gumbel_cdf(5, 0.5, 2), 0.5, 2))

def sample_gumbel(shape):
    p = np.random.random(shape)
    return inv_gumbel_cdf(p)

def sample_with_gumbel_noise(logits, size):
    noise = sample_gumbel((size, len(logits)))
    return np.argmax(logits + noise, axis=1)

np.random.seed(1111)
logits = (np.random.random(10) - 0.5) * 2  # (-1, 1)

pop = 100000
softmax_samples = sample_with_softmax(logits, pop)
gumbel_samples = sample_with_gumbel_noise(logits, pop)

plt.subplot(1, 2, 1)
plt.hist(softmax_samples)

plt.subplot(1, 2, 2)
plt.hist(gumbel_samples)
plt.show()