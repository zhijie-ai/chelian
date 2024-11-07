# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/5 17:36
# @User   : RANCHODZU
# @File   : dp03.py
# @e-mail : zushoujie@ghgame.cn
from typing import List


# 代码随想录中的版本https://programmercarl.com/0377.%E7%BB%84%E5%90%88%E6%80%BB%E5%92%8C%E2%85%A3.html#%E6%80%9D%E8%B7%AF
# 相当于是完全背包中的二维数组的解法
def combinationSum4(nums: List[int], target: int) -> int:
    # dp[][j]和为j的组合的总数
    dp = [[0] * (target + 1) for _ in nums]

    for i in range(len(nums)):
        dp[i][0] = 1

    # 这里不能初始化dp[0][j]。dp[0][j]的值依赖于dp[-1][j-nums[0]]

    for j in range(1, target + 1):
        for i in range(len(nums)):

            if j - nums[i] >= 0:
                dp[i][j] = (
                    # 不放nums[i]
                    # i = 0 时，dp[-1][j]恰好为0，所以没有特殊处理
                        dp[i - 1][j] +
                        # 放nums[i]。对于和为j的组合，只有试过全部物品，才能知道有几种组合方式。所以取最后一个物品dp[-1][j-nums[i]]
                        dp[-1][j - nums[i]]
                )
            else:
                dp[i][j] = dp[i - 1][j]
    print(dp)
    return dp[-1][-1]


def dp01(nums, target):
    dp = [[0] * (target + 1) for _ in nums]

    for i in range(len(nums)):
        dp[i][0] = 1
    for j in range(target + 1):
        dp[0][j] = 1

    for i in range(len(nums)):
        for j in range(target+1):
            if j - nums[i] < 0:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] += dp[i][j-nums[i]]
    print(dp)


if __name__ == '__main__':
    nums = [1, 2, 3]
    target = 4
    combinationSum4(nums, target)
    dp01(nums, target)