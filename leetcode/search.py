# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/9/19 9:08                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------

#左闭右闭区间[left, right]
def search(nums, target):
    if len(nums)==0:
        return -1

    left, right = 0, len(nums)-1
    while left <= right:
        middle = (right+left)//2
        middle = left + ((right-left)>>1)
        if nums[middle] < target:
            left = middle + 1
        elif nums[middle]>target:
            right = middle-1
        else:
            return middle
    return -1

# 左闭右开区间[left,right)
def search_v2(nums, target):
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums)
    while left < right:
        mid = left + ((right-left)>>1)
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    return -1

if __name__ == '__main__':
    nums = [1,2,3,4,5,6,7,8,9]
    target = 4
    print(search(nums, target))