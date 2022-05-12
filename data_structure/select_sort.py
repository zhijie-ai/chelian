#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/5/3 7:36                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# 选择排序：首先在未排序的数组中找出最大(or 最小)元素，然后将其存放到数列的起始位置；接着，再从剩余未排序的元素中国继续
#   寻找最大(or 最小)元素，然后放到一直排序列表序列的末尾。以此类推，直到所有元素均排序完毕

def select_sort(arr):
    for i in range(len(arr)):#有序区间的末尾位置
        min = i
        for j in range(i+1,len(arr)):#无序区间的起始位置
            if arr[j]<arr[min]:
                min = j#找出最小元素下标

        if min != i:#新的最小元素诞生了
            arr[i],arr[min] = arr[min],arr[j]# 交换2个元素的位置

def select_sort2(arr):
    for i in range(len(arr)):
        min = i
        for j in range(i+1,len(arr)):
            if arr[j]< arr[min]:
                min = j

        if min!=j:
            arr[i], arr[min] = arr[min], arr[i]


