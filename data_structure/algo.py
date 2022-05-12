# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/5/12 21:37                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------

def bubble(arr):
    for i in range(len(arr))[::-1]:
        flag=0
        for j in range(i):
            if arr[j]>arr[j+1]:
                arr[j],arr[j+1] = arr[j+1],arr[j]
                flag=1
        if not flag:
            break
    return arr

def quick(arr,l,r):
    if l<r:
        i=l
        j=r
        m=arr[i]
        while i<j:
            while i<j and arr[j]>m:
                j-=1
            if i<j:
                arr[i] = arr[j]
                i+=1
            while i<j and arr[i]< m:
                i+=1
            if i<j:
                arr[j]=arr[i]
                j-=1

        arr[i] = m
        quick(arr,l,i-1)
        quick(arr,i+1,r)

def quick2(arr):
    if len(arr)<2:
        return arr

    base = arr[0]
    left = [e for e in arr[1:] if e<=base]
    right =[e for e in arr[1:] if e>base]
    return quick2(left+[base]+right)

def insert(arr):
    for i in range(1,len(arr)):
        for j in range(i)[::-1]:
            if arr[j]<arr[i]:
                break

        if j!=i-1:
            tmp = arr[i]
            for k in range(j,i)[::-1]:
                arr[k+1] = arr[k]
            arr[k+1] = tmp
    return arr

def insert2(arr):
    for i in range(1,len(arr)):
        j=i-1
        k = arr[i]
        while j>0:
            if arr[j]>k:
                arr[j+1] = arr[j]
                arr[j] = k
            j-=1
    return arr

def select(arr):
    for i in range(len(arr)):
        min = i
        for j in range(i+1,len(arr)):
            if arr[j]<arr[min]:
                min=j

        if min!=i:
            arr[i],arr[min] = arr[min],arr[i]



