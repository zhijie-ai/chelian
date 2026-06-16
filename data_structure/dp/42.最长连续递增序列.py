# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2026/1/16 18:08                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import numpy as np

def dp01(nums):
    if len(nums) == 0:
        return 0
    result = 1
    dp = [1] * len(nums)
    for i in range(len(nums)-1):
        if nums[i+1] > nums[i]: #连续记录
            dp[i+1] = dp[i] + 1
        result = max(result, dp[i+1])

    print(dp)
    print(result)
    return result

def dp02(s):
    l=r=0
    result = 0
    for i in range(0,len(s)-1):
        if s[i+1] >= s[i]:
            r += 1
        else:
            result = max(result, r-l+1)
            l = i+1
            r = i+1
    result = max(result, r-l+1)

    print(result)


if __name__ == '__main__':
    s = [1, 3, 5, 4, 7, 8,9,11,15]
    s = np.random.randint(0,100, 20)
    print(s)
    dp01(s)
    dp02(s)