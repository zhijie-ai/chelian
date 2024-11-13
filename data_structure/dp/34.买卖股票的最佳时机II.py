# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/12 16:04
# @User   : RANCHODZU
# @File   : 34.买卖股票的最佳时机II.py
# @e-mail : zushoujie@ghgame.cn
prices = [7,1,5,3,6,4]

# 能买卖多次
def dp01():
    if len(prices) == 0:
        return 0

    # dp[i][0] 表示第i天持有股票所得现金。
    # dp[i][1] 表示第i天不持有股票所得最多现金
    dp = [[0]*2 for _ in range(len(prices))]

    dp[0][0] = -prices[0]
    dp[0][1] = 0

    for i in range(1, len(prices)):
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] - prices[i])
        dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i])

    print(dp[-1][1])


if __name__ == '__main__':
    dp01()