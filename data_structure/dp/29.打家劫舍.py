# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/11 11:07
# @User   : RANCHODZU
# @File   : 29.打家劫舍.py
# @e-mail : zushoujie@ghgame.cn

rob = [2, 7, 9, 3, 1]
def dp1d(nums):
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

def dp2d(nums):
    if not nums:
        return 0
    n = len(nums)
    # 创建二维动态规划数组，dp[i][0]表示不抢劫第i个房屋的最大金额，dp[i][1]表示抢劫第i个房屋的最大金额
    dp = [[0, 0] for _ in range(n)]
    dp[0][1] = nums[0]

    for i in range(1, n):
        # 不抢劫第i个房屋，最大金额为前一个房屋抢劫和不抢劫的最大值
        dp[i][0] = max(dp[i-1][0], dp[i-1][1])
        # # 抢劫第i个房屋，最大金额为前一个房屋不抢劫的最大金额加上当前房屋的金额
        dp[i][1] = dp[i-1][0] + nums[i]

    print(dp)
    return max(dp[n - 1][0], dp[n - 1][1])  # 返回最后一个房屋中可抢劫的最大金额


def dp_optimization(nums):
    if not nums:  # 如果没有房屋，返回0
        return 0

    prev_max = 0  # 上一个房屋的最大金额
    curr_max = 0  # 当前房屋的最大金额

    for num in nums:
        temp = curr_max  # 临时变量保存当前房屋的最大金额
        curr_max = max(prev_max + num, curr_max)  # 更新当前房屋的最大金额
        prev_max = temp  # 更新上一个房屋的最大金额

    print(curr_max)
    return curr_max  # 返回最后一个房屋中可抢劫的最大金额


if __name__ == '__main__':
    dp1d(rob)
    dp2d(rob)
    dp_optimization(rob)