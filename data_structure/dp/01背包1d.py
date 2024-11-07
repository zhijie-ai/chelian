# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/1 11:29
# @User   : RANCHODZU
# @File   : 01背包1d.py
# @e-mail : zushoujie@ghgame.cn
"""
01背包有2个问题，一个是max问题，一个是组合数的问题
"""

# 01背包先物品再背包和先背包再物品在二维情况下是没区别的，一维情况下要先物品再背包，且背包是逆序
# 01背包问题
def dp01_1d():
    n, bagweight = 3, 4
    weight = [1, 3, 4]
    value = [15, 20, 30]

    dp = [0] * (bagweight + 1)  # 创建一个动态规划数组dp，初始值为0

    dp[0] = 0  # 初始化dp[0] = 0,背包容量为0，价值最大为0

    for i in range(n):  # 应该先遍历物品，如果遍历背包容量放在上一层，那么每个dp[j]就只会放入一个物品
        for j in range(bagweight, weight[i] - 1, -1):  # 倒序遍历背包容量是为了保证物品i只被放入一次
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i])

    print(dp)  # [0, 15, 15, 20, 35]


def knapsack_with_items(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    prev = [[-1] * (capacity + 1) for _ in range(n + 1)]

    # 填充dp和prev数组
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if w >= weights[i - 1]:
                if dp[i - 1][w] < dp[i - 1][w - weights[i - 1]] + values[i - 1]:
                    dp[i][w] = dp[i - 1][w - weights[i - 1]] + values[i - 1]
                    prev[i][w] = 1  # 表示选择了第i个物品
                else:
                    dp[i][w] = dp[i - 1][w]
                    prev[i][w] = -1  # 表示没有选择第i个物品
            else:
                dp[i][w] = dp[i - 1][w]
                prev[i][w] = -1

                # 找到最大价值及其对应的背包容量
    max_value = max(dp[n])
    max_w = dp[n].index(max_value)  # 这里假设背包容量足够大，总能找到最大价值

    # 回溯路径以确定所选物品
    selected_items = []
    w = max_w
    for i in range(n, 0, -1):
        if prev[i][w] == 1:
            selected_items.append(i - 1)  # 物品索引从0开始，所以减1
            w -= weights[i - 1]
    selected_items.reverse()  # 因为是从后往前回溯的，所以需要反转列表

    return max_value, selected_items


def dp01_1d_():
    n, bagweight = 3, 4
    weight = [1, 3, 4]
    value = [15, 20, 30]

    dp = [0] * (bagweight + 1)  # 创建一个动态规划数组dp，初始值为0

    dp[0] = 0  # 初始化dp[0] = 0,背包容量为0，价值最大为0

    for i in range(n)[::-1]:  # 应该先遍历物品，如果遍历背包容量放在上一层，那么每个dp[j]就只会放入一个物品，物品逆序
        for j in range(bagweight, weight[i] - 1, -1):  # 倒序遍历背包容量是为了保证物品i只被放入一次
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i])

    print(dp)  # [0, 15, 15, 20, 35]


def dp03():
    nums = [1, 1, 1, 1, 1]
    target = 3
    total_sum = sum(nums)  # 计算nums的总和
    if abs(target) > total_sum:
        return 0  # 此时没有方案
    if (target + total_sum) % 2 == 1:
        return 0  # 此时没有方案
    target_sum = (target + total_sum) // 2  # 目标和
    dp = [0] * (target_sum + 1)  # 创建动态规划数组，初始化为0
    dp[0] = 1  # 当目标和为0时，只有一种方案，即什么都不选
    for num in nums:
        for j in range(target_sum, num - 1, -1):
            dp[j] += dp[j - num]  # 状态转移方程，累加不同选择方式的数量
    return dp  # 返回达到目标和的方案数


if __name__ == '__main__':
    # dp01_1d()
    # 示例数据
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 5

    # 计算最大价值及所选物品
    max_val, items = knapsack_with_items(weights, values, capacity)
    print(f"最大价值: {max_val}")
    print(f"所选物品索引: {items}")
    print(dp03())
