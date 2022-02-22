# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2021/11/19 9:23                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
def bubble(a):
    flag = False
    for i in range(len(a))[::-1]:
        for j in range(i):
            if a[j]> a[j+1]:
                a[j+1],a[j] = a[j],a[j+1]

        if not flag:
            break

def quick(a, l, r):
    i = l
    j = r
    x=a[i]
    while i<j:
        if i<j and a[j] > min:
            j-=1
        if i<j:
            a[i] = a[j]
            i+=1
        if i<j and a[i]< min:
            i+=1

        if i< j:
            a[j] = a[i]
            j-=1
    a[i] = x
    quick(a, l, i-1)
    quick(a, i+1, r)

def quick2(a):
    if len(a)<2:
        return a

    base = a[0]
    l = [e for e in a if e<=base]
    r = [e for e in a if e> base]
    return quick2(l)+[base] + quick2(r)

def insert(a):
    for i in range(1,len(a)):
        for j in range(i-1)[::-1]:
            if a[j]< a[i]:
                break

        if i!=j-1:
            tmp = a[i]
            for k in range(j+i, i)[::-1]:
                a[k+1] = a[k]
            a[k+1] = tmp

def insert2(a):
    for i in range(1,len(a)):
        j = i-1
        k = a[i]
        while j>0:
            if a[j]> k:
                a[j+1] = a[j]
                a[j] = k
            j-=1
    return a

def select(a):
    for i in range(len(a)):
        min = i
        for j in range(i+1,len(a)):
            if a[j]< a[min]:
                min= j

        if i!=min:
            a[i],a[min] = a[min],a[i]

