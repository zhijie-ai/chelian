# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/13 17:00
# @User   : RANCHODZU
# @File   : 36.买卖股票的最佳时机IV.py
# @e-mail : zushoujie@ghgame.cn
prices = [3,2,6,5,0,3]
k = 2

# 最多k次
def dp01():
    if len(prices) == 0:
        return 0
    dp = [[0] * (2 * k + 1) for _ in range(len(prices))]
    for j in range(1, 2 * k, 2):
        dp[0][j] = -prices[0]
    for i in range(1, len(prices)):
        for j in range(0, 2 * k - 1, 2):
            dp[i][j + 1] = max(dp[i - 1][j + 1], dp[i - 1][j] - prices[i])
            dp[i][j + 2] = max(dp[i - 1][j + 2], dp[i - 1][j + 1] + prices[i])

    print(dp)
    return dp[-1][2 * k]

def dp02():
    if len(prices) == 0:
        return 0
    dp = [0] * (2 * k + 1)
    for i in range(1, 2 * k, 2):
        dp[i] = -prices[0]
    for i in range(1, len(prices)):
        for j in range(1, 2 * k + 1):
            if j % 2:
                dp[j] = max(dp[j], dp[j - 1] - prices[i])
            else:
                dp[j] = max(dp[j], dp[j - 1] + prices[i])
    print(dp)
    return dp[2 * k]


if __name__ == '__main__':
    dp01()
    dp02()
