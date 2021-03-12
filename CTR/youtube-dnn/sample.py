#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/1/20 16:08                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
def sample(n, cnt):
    a = i = 0
    table = []
    prob = 0
    z = sum(num ** 0.75 for num in cnt) # denominator

    prob = cnt[i] ** 0.75 / z # cumulative probability
    for a in range(n):
        '''
        loop invariant:
            at any time, a <= prob * n,
            with the same i, largest a = prob * n,
                             smallest a = prob_old * n + 1,
                             largest a - smallest a + 1 = prob * n - prob_old * n, 
            which is the expectation of count of i.
        '''
        table.append(i)
        if a > prob * n:
            i += 1
            prob += cnt[i] ** 0.75 / z

    return table


if __name__ == '__main__':
    from collections import Counter

    # count, aka frequency
    cnt = [1, 2, 3, 100, 15]
    # 0.75 power
    prob = [x ** 0.75 for x in cnt]
    prob = [x / sum(prob) for x in prob]
    print(prob)

    # sampling
    res = Counter(sample(50000, cnt))
    print([x / sum(res.values()) for x in res.values()])