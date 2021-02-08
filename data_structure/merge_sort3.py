#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/5/8 8:35                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

def merge(left,right):
    c = []
    i = j = 0
    while i<len(left) and j<len(right):
        if left[i]<right[j]:
            c.append(left[i])
            i+=1
        else:
            c.append(right[j])
            j+=1

    if i<len(left):
        c.append(left[i:])

    if j<len(right):
        c.append(right[j:])

    return c

def merge_sort_up2down(arr):
    if len(arr)<=1:
        return

    mid = len(arr)//2
    left = merge_sort_up2down(arr[:mid])
    right = merge_sort_up2down(arr[mid:])
    return merge(left,right)