# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/12 16:45
# @User   : RANCHODZU
# @File   : 35.买卖股票的最佳时机III.py
# @e-mail : zushoujie@ghgame.cn
prices = [7,1,5,3,6,4]
def dp01():
    if len(prices) == 0:
        return 0

    """
    0. 没有操作 （其实我们也可以不设置这个状态）
    1. 第一次持有股票
    2. 第一次不持有股票
    3. 第二次持有股票
    4. 第二次不持有股票
    """
    dp = [[0] * 5 for _ in range(len(prices))]

    dp[0][1] = -prices[0]
    dp[0][2] = 0
    dp[0][3] = -prices[0]
    dp[0][4] = 0

    for i in range(1, len(prices)):
        dp[i][0] = dp[i - 1][0]
        dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
        dp[i][2] = max(dp[i-1][2], dp[i-1][1] + prices[i])
        dp[i][3] = max(dp[i-1][3], dp[i-1][2] - prices[i])
        dp[i][4] = max(dp[i-1][4], dp[i-1][3] + prices[i])

    # print(dp[-1][4])
    print(dp[1:])

def dp01_():
    if len(prices) == 0:
        return 0
    """
    0. 第一次持有股票
    1. 第一次不持有股票
    2. 第二次持有股票
    3. 第二次不持有股票
    """
    dp = [[0] * 4 for _ in range(len(prices))]

    dp[0][0] = -prices[0]
    dp[0][1] = 0
    dp[0][2] = -prices[0]
    dp[0][3] = 0

    for i in range(1, len(prices)):
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] - prices[i])
        dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i])
        dp[i][2] = max(dp[i-1][2], dp[i-1][3] - prices[i])
        dp[i][3] = max(dp[i-1][3], dp[i-1][2] + prices[i])

    # print(dp[-1][3])
    print(dp)

def dp02():
    if len(prices) == 0:
        return 0
    dp = [0] * 5
    dp[1] = -prices[0]
    dp[3] = -prices[0]
    for i in range(1, len(prices)):
        dp[1] = max(dp[1], dp[0] - prices[i])
        dp[2] = max(dp[2], dp[1] + prices[i])
        dp[3] = max(dp[3], dp[2] - prices[i])
        dp[4] = max(dp[4], dp[3] + prices[i])
    print(dp)
    return dp[4]


if __name__ == '__main__':
    dp01()
    dp01_()
    dp02()
