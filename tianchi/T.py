#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/1 20:46                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import random

def get_num(arr):
    sum = 0
    temp = []
    for i in range(55):
        num = random.choice(arr)
        sum+=num
        temp.append(num)
        if sum == 55:
            temp = sorted(temp)
            return True,tuple(temp)

    return False,None

def times_():
    for i in range(100000):
        flag, num = get_num(arr)
        if flag:
            res.add(num)
            return len(res)
    return 0

arr = [1,3,5,7,9,11,13]
res = set()
times = set()
for i in range(10):
    for i in range(100000):
        flag, num = get_num(arr)
        if flag:
            res.add(num)
            times.add(len(res))


print(times)




