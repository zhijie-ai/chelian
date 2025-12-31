# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2025/12/30 16:05                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
def vali_difference():
    coins = [2, 1, 5]  # 注意：2在1前面！
    amount = 3

    # 方法1：先物品后背包
    dp1 = [[0]*(amount+1) for _ in range(len(coins))]
    for i in range(len(coins)):
        dp1[i][0] = 1
    for j in range(coins[0], amount+1):
        dp1[0][j] += dp1[0][j-coins[0]]

    for i in range(1, len(coins)):
        for j in range(1, amount+1):
            if j < coins[i]:
                dp1[i][j] = dp1[i-1][j]
            else:
                dp1[i][j] = dp1[i][j-coins[i]] + dp1[i-1][j]

    # 方法2：先背包后物品
    dp2 = [[0]*(amount+1) for _ in range(len(coins))]
    for i in range(len(coins)):
        dp2[i][0] = 1
    for j in range(coins[0], amount+1):
        dp2[0][j] += dp2[0][j-coins[0]]

    for j in range(1, amount+1):
        for i in range(1, len(coins)):
            if j < coins[i]:
                dp2[i][j] = dp2[i-1][j]
            else:
                dp2[i][j] = dp2[i][j-coins[i]] + dp2[i-1][j]

    print("先物品后背包:")
    for row in dp1:
        print(row)
    print("\n先背包后物品:")
    for row in dp2:
        print(row)

def dp01(n, bag_weight, weight, value):
    dp = [[0] * (bag_weight + 1) for _ in range(n)]

    # 初始化
    for j in range(weight[0], bag_weight + 1):
        dp[0][j] = dp[0][j - weight[0]] + value[0]

    # 动态规划
    for i in range(1, n):
        for j in range(bag_weight + 1):
            if j < weight[i]:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - weight[i]] + value[i])

    return dp

def dp02(n, bag_weight, weight, value):
    dp = [[0] * (bag_weight + 1) for _ in range(n)]

    # 初始化
    for j in range(weight[0], bag_weight + 1):
        dp[0][j] = dp[0][j - weight[0]] + value[0]

    # 动态规划
    for j in range(bag_weight + 1):
        for i in range(1, n):
            if j < weight[i]:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - weight[i]] + value[i])

    return dp


if __name__ == '__main__':
    # vali_difference()
    weight = [1, 4, 3]  # 注意：2在1前面！
    value = [15, 20, 30]  # 注意：2在1前面！
    n = 3
    bag_weight=4


    print(dp01(n, bag_weight, weight, value))
    print(dp02(n, bag_weight, weight, value))