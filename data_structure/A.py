# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/5/12 22:22                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
def quick(a,l,r):
    if l<r:
        i=l
        j=r
        x=a[i]
        while i<j:
            while i<j and a[j]>x:
                j-=1
            if i<j:
                a[i]=a[j]
                i+=1
            while i<j and a[i]<x:
                i-=1
            if i<j:
                a[j]=a[i]
                j-=1

        a[i]=x
        quick(a,l,i-1)
        quick(a,i+1,r)

def bubble(a):
    for i in range(len(a))[::-1]:
        flag=0
        for j in range(i):
            if a[j]<a[j+1]:
                a[j],a[j+1] = a[j+1],a[j]
                flag=1
        if not flag:
            break

def insert(a):
    for i in range(1,len(a)):
        for j in range(i)[::-1]:
            if a[j]<a[i]:
                break
        if j!=i-1:
            tmp = a[i]
            for k in range(j+1,i)[::-1]:
                a[k+1]=a[k]
            a[k] = tmp # 这种情况是对的，其他情况不对，注意原始C代码的区别。不能是k+1

def select(a):
    for i in range(len(a)):
        min = a[i]
        for j in range(i+1,len(a)):
            if a[j]<a[min]:
                min=j
        if i!=min:
            a[i],a[min]=a[min],a[i]

def quick2(arr):
    if len(arr)<2:
        return arr
    base = arr[0]
    left = [e for e in arr[1:] if e<=base]
    right = [e for e in arr[1:] if e>base]
    return quick2(left+[base]+right)

def insert2(arr):
    for i in range(1,len(arr)):
        j = i-1
        x=arr[i]
        while j>0:
            if arr[j]>x:
                arr[j+1]=arr[j]
                arr[j]=x
            j-=1

