# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/7 15:28
# @User   : RANCHODZU
# @File   : 23.零钱兑换.py
# @e-mail : zushoujie@ghgame.cn

coins = [1, 2, 5]
amount = 5


def dp1d():
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in coins:
        for j in range(i, amount + 1):
            dp[j] = min(dp[j], dp[j-i] + 1)
    print(dp)
    return dp[amount] if dp[amount] != float('inf') else -1


def dp2d():
    from utils import find_selected_item_v2
    dp = [[float('inf')] * (amount + 1) for _ in range(len(coins))]

    for j in range(amount + 1):
        if j >= coins[0]:
            dp[0][j] = j
        else:
            dp[0][j] = 0

    for i in range(1, len(coins)):
        for j in range(amount + 1):
            if j >= coins[i]:
                dp[i][j] = min(dp[i-1][j], dp[i][j - coins[i]] + 1)
            else:
                dp[i][j] = dp[i-1][j]

    print(dp)


def dp2d_():
    # 前i个硬币凑成金额为j的货币组合数为dp[i][j]
    dp = [[float('inf')] * (amount + 1) for _ in range(len(coins)+1)]
    dp[0][0] = 0  # 凑成金额0需要0个硬币
    for i in range(1, len(coins)+1):
        for j in range(amount+1):
            if j >= coins[i-1]:
                dp[i][j] = min(dp[i-1][j], dp[i][j - coins[i-1]] + 1)
            else:
                dp[i][j] = dp[i-1][j]

    print(dp[1:])


if __name__ == '__main__':
    dp1d()
    dp2d()
    dp2d_()
