# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/5 18:26
# @User   : RANCHODZU
# @File   : 01背包2d.py
# @e-mail : zushoujie@ghgame.cn

"""
01背包有2个问题，一个是max问题，一个是组合数的问题
注意dp03中的定义和dp04的区别，03中的i是类别，表示在考虑前 i 种物品，且背包容量为 j 的情况下的组合数
"""
# 01背包在一维情况下要先物品再背包，且背包是逆序
def dp01():
    n, bagweight = 3, 4

    weight = [1, 3, 4]
    value = [15, 20, 30]

    dp = [[0] * (bagweight + 1) for _ in range(n)]

    for j in range(weight[0], bagweight + 1):
        dp[0][j] = value[0]

    for i in range(1, n):
        for j in range(bagweight + 1):
            if j < weight[i]:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])

    print(dp)


def dp02():
    n, bagweight = 3, 4

    weight = [1, 3, 4]
    value = [15, 20, 30]

    dp = [[0] * (bagweight + 1) for _ in range(n)]

    for j in range(weight[0], bagweight + 1):
        dp[0][j] = value[0]

    for j in range(bagweight + 1):
        for i in range(1, n):
            if j < weight[i]:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])

    print(dp)


def dp02_():
    n, bagweight = 3, 4

    weight = [1, 3, 4]
    value = [15, 20, 30]

    dp = [[0] * (bagweight + 1) for _ in range(n+1)]
    dp[0][0] = 0

    for j in range(1, bagweight + 1):
        for i in range(1, n+1):
            if j < weight[i-1]:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i-1]] + value[i-1])

    print(dp[1:])


# 组合数的01背包问题
def dp03():
    nums = [1, 1, 1, 1, 1]
    target = 3
    total_sum = sum(nums)  # 计算nums的总和
    if abs(target) > total_sum:
        return 0  # 此时没有方案
    if (target + total_sum) % 2 == 1:
        return 0  # 此时没有方案
    target_sum = (target + total_sum) // 2  # 目标和

    # 创建二维动态规划数组，行表示选取的元素数量，列表示累加和
    dp = [[0] * (target_sum + 1) for _ in range(len(nums) + 1)]

    # 初始化状态
    dp[0][0] = 1

    # 动态规划过程
    for i in range(1, len(nums) + 1):
        for j in range(target_sum + 1):
            dp[i][j] = dp[i - 1][j]  # 不选取当前元素
            if j >= nums[i - 1]:
                dp[i][j] += dp[i - 1][j - nums[i - 1]]  # 选取当前元素
    print(dp[1:])  # [1, 5, 10, 10, 5]


# 和dp03的区别在于，二维数组的维度和初始化
def dp04():
    nums = [1, 1, 1, 1, 1]
    target = 3
    total_sum = sum(nums)  # 计算nums的总和
    if abs(target) > total_sum:
        return 0  # 此时没有方案
    if (target + total_sum) % 2 == 1:
        return 0  # 此时没有方案
    target_sum = (target + total_sum) // 2  # 目标和

    # 创建二维动态规划数组，行表示选取的元素数量，列表示累加和
    dp = [[0] * (target_sum + 1) for _ in range(len(nums))]
    for i in range(len(nums)):
        dp[i][0] = 1

    for i in range(target_sum + 1):
        if i <= 1:
            dp[0][i] = 1

    # 动态规划过程
    for i in range(1, len(nums)):
        for j in range(1, target_sum + 1):
            dp[i][j] = dp[i - 1][j]  # 不选取当前元素
            if j >= nums[i - 1]:
                dp[i][j] += dp[i - 1][j - nums[i - 1]]  # 选取当前元素
    print(dp) # [1, 5, 10, 10, 5]


if __name__ == '__main__':
    dp01()
    dp02()
    dp02_()
    dp03()
    dp04()