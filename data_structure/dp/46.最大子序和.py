# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/15 15:32
# @User   : RANCHODZU
# @File   : 46.最大子序和.py
# @e-mail : zushoujie@ghgame.cn
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]

def dp01():
    # 包括下标i（以nums[i]为结尾）的最大连续子序列和为dp[i]
    dp = [0] * (len(nums))
    dp[0] = nums[0]

    for i in range(1, len(nums)):
        dp[i] = max(dp[i-1]+nums[i], nums[i])

    print(dp)

def dp02():
    ind = [[0, 0] for _ in range(len(nums)+1)]
    # 前i个数的连续子序列最大和
    dp = [0] * (len(nums)+1)

    for i in range(1, len(nums)+1):
        if dp[i-1] + nums[i-1] >= nums[i-1]:
            dp[i] = dp[i-1] + nums[i-1]
            ind[i][0] = ind[i-1][0]
            ind[i][1] = i-1
        else:
            dp[i] = nums[i-1]
            ind[i][0] = i-1
            ind[i][1] = i-1
    print(ind[1:])
    print(dp[1:])  # [0, -2, 1, -2, 4, 3, 5, 6, 1, 5]


if __name__ == '__main__':
    dp01()
    dp02()