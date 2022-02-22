# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2021/11/18 21:11                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import sys
for line in sys.stdin:
    a = line.split()
    print(int(a[0]) + int(a[1]))

def bubble(a):
    flag = 0
    for i in range(len(a)-1, 0, -1):
        for j in range(i):
            if a[j]>a[j+1]:
                a[j], a[j+1] = a[j], a[j+1]
                flag=1

        if not flag:
            break

def insert(a):
    for i in range(1,len(a)):
        for j in range(i-1,0, -1):
            if a[i]> a[j]:
                break

        if j != i-1:
            tmp = a[i]
            for k in range(j+1, i)[::-1]:
                a[k+1] = a[k]
            a[k+1] = tmp

def insert2(a):
    for i in range(1,len(a)):
        key = a[i]
        j = i-1
        while j>0:
            if a[j]> key:
                a[j+1] = a[j]
                a[j]=key
            j-=1
    return a

def quick_sort(a,l,r):
    i = l
    j = r
    x = a[i]
    while i<j:
        while i< j and a[j]> x:
            j-=1
        if i<j:
            a[i] = a[j]
            i+=1
        while i<j and a[i]<x:
            i+=1
        if i<j:
            a[j] = a[i]
            j-=1
    a[i] = x
    quick_sort(a,l,i-1)
    quick_sort(a,i+1,r)

def quick_sort2(a):
    if len(a) <2:
        return a

    base = a[0]
    l = [i for i in a if i <= base]
    r = [i for i in a if i > base]
    return quick_sort2(l)+[base] +quick_sort2(r)

def select(a):
    for i in range(len(a)):
        min = i
        for j in range(i+1,len(a)):
            if a[j] < a[min]:
                min = j

        if min != i:
            a[min], a[i] = a[i], a[min]



