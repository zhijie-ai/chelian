#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/1/26 10:03                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import numpy as np

'''
根据权重
1. nlp中subsample: 根据概率来保留，目前似乎还不知道怎么做,知道了，参考总结中的描述(笔记.txt，1009行)
2. 如果sum(w)=1,则可以根据np.random.choice来采样，就算概率很小也可以，也可以按照word2vec源码中的负采样的思路来采样
3. 如果sum(w)!=1,既可以归一化操作让概率和为一，也可以采用R^(1/w)的思路来采样。
'''

# 样本加权采样方式
def sample(sample={'A':0.6,'B':0.7,'C':0.2}):
    dit = {}
    for key,w in sample.items():
        R = np.random.sample()
        S = R**(1/w)
        dit[key]=S
    dit = sorted(dit.items(),key=lambda x:x[1],reverse=True)[0][0]
    return dit

# 指数采样方式
def exp_sample(sample={'A':0.4,'B':0.1,'C':0.5}):
    dit = {}
    for key,w in sample.items():
        S = np.random.exponential(w,1)
        dit[key]=S
    dit = sorted(dit.items(),key=lambda x:x[1],reverse=True)[0][0]
    return dit

def simulation(n=10000):
    from collections import Counter
    L = []
    for i in range(n):
        L.append(exp_sample())
    cnt = Counter(L)
    return cnt

# genism中的降采样，似乎不是这样实现的
def subsample(word_probability=1):
    sample_int = int(round(word_probability * 2**32))

    next_random = (2**24) * np.random.randint(0, 1) + np.random.randint(0, 2**24)
    random_int32 = (next_random * 25214903917 + 11) & 281474976710655
    print(sample_int)
    print(random_int32)
    return sample_int<random_int32


def nagative_sample():
    np.random.seed(1126)
    def gen_data():
        num = []
        for i in range(100000):
            high = np.random.randint(10,100000)
            n = np.random.randint(1,100000)
            num.append(n)
        return np.array(num,dtype=np.float)

    num = gen_data()
    w = num/np.sum(num)
    print(np.sum(w))
    print(np.min(w),np.max(w))
    print(np.random.choice(range(100000),p=w,size=(1000),replace=False))


# word2vec中的负采样
def nagetive_sample(n, cnt):
    a = i = 0
    table = []
    prob = 0
    z = sum(num ** 0.75 for num in cnt)

    prob = cnt[i] ** 0.75/z
    for a in range(n):
        table.append(i)
        if a > prob * n:
            i += 1
            prob += cnt[i] ** 0.75 / z
    return table


def negative_sample_(n, cnt):
    z = sum(num ** 0.75 for num in cnt)
    cumulative_prob = 0
    probabilities = [num ** 0.75 / z for num in cnt]
    cumulative_probabilities = [cumulative_prob + prob for prob in probabilities]
    table = []

    for a in range(n):
        # 找到第一个累积概率大于或等于随机数的索引

        i = next(i for i, cp in enumerate(cumulative_probabilities) if cp > random.random())

        table.append(i)

    return table


import random



# print(subsample())

# print(simulation())
# nagative_sample()

if __name__ == '__main__':
    from collections import Counter

    # count, aka frequency
    cnt = [1, 2, 3, 100, 15]
    # 0.75 power
    prob = [x ** 0.75 for x in cnt]
    prob = [x / sum(prob) for x in prob]
    print(prob, sum(prob))
    p = [x / sum(cnt) for x in cnt]
    p = [x ** 0.75 for x in p]
    p_ = [x / sum(p) for x in p]
    print(p_, sum(p_))

    # sampling
    tab = nagetive_sample(50000, cnt)
    # print(tab)
    res = Counter(tab)
    print(res)
    print([x / sum(res.values()) for x in res.values()])
    ind = np.random.randint(0,50000,10)
    print(ind)
    print(np.array(tab)[ind])