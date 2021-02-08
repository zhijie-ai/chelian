#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/5/2 22:11                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# 直接插入排序
#把待排序的数组看成一个有序表和无序表，开始时有序表中只有一个元素，无序表中有n-1个元素。排序过程每次从
#   无序表中取出第一个元素，将他插入到有序表中适当位置。使之成为新的有序表，重复n-1次可完成排序过程

def insert_sort(arr):
    for i in range(1,len(arr)):# 无序
        for j in list(range(0,i))[::-1]:#有序
            if arr[j]<arr[i]:
                break# 找到了插入位置

        # 如果找到了一个插入位置
        if j != i-1:
            temp = arr[i]
            for k in list(range(j+1,i))[::-1]:
                arr[k+1] = arr[k]
            arr[k+1] = temp

def insert_sort2(arr):
    count = len(arr)
    for i in range(1,count):
        key = arr[i]
        j = i-1
        while j>=0:
            if arr[j]>key:
                arr[j+1] = arr[j]
                arr[j] = key
            j -=1
    return arr
