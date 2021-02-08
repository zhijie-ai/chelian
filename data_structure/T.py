#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/4/29 16:21                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np
data = np.random.randint(0,100,20)
print(data[0:20])
temp = data[0]
data[0]=data[1]
data[1] = temp
print(data)
def name(data,age,salary):
    i = age
    print(i)
    print(data)
    print(age)
    print(salary)

# name(data,20,20)

arr = np.random.randint(0,1000000000,1000000000)

min = 0
for i in range(1,len(arr)):
    if arr[i]<arr[min]:
        min = i

print('{}:{}'.format(min,arr[min]))