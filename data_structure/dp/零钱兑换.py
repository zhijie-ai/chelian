# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/7 15:28
# @User   : RANCHODZU
# @File   : 零钱兑换.py
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


if __name__ == '__main__':
    dp1d()
    dp2d()
