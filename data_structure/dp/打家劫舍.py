# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/11 11:07
# @User   : RANCHODZU
# @File   : 打家劫舍.py
# @e-mail : zushoujie@ghgame.cn

nums = [2, 7, 9, 3, 1]
def dp1d():
    if nums == 0:
        return 0
    if len(nums) == 1:
        return nums[0]

    dp = [0] * len(nums)
    dp[0] = nums[0]
    dp[1] = max(dp[0], nums[1])

    for i in range(2, len(nums)):
        # 对于每个房屋，选择抢劫当前房屋和抢劫前一个房屋的最大金额
        dp[i] = max(dp[i-2]+nums[i], dp[i-1])

    print(dp)


if __name__ == '__main__':
    dp1d()