#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/5/3 22:16                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# 桶排序

def bucket_sort(arr,max):
    buckets = [0]*max

    # 计数
    for i in arr:
        buckets[i]+=1

    # 排序
    j=0
    for i in range(max):
        while buckets[i]>0:
            buckets[i] -=1
            arr[j]=i
            j += 1


arr = [8,2,3,4,3,6,6,3,9]
print('before sort:',arr)
bucket_sort(arr,10)
print('after sort:',arr)


