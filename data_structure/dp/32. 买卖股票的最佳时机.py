# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/12 10:28
# @User   : RANCHODZU
# @File   : 32. 买卖股票的最佳时机.py
# @e-mail : zushoujie@ghgame.cn

prices = [7,1,5,3,6,4]
# 只能买卖一次
def dp01():
    if len(prices) == 0:
        return 0

    dp = [[0]*2 for _ in range(len(prices))]

    dp[0][0] = -prices[0]
    dp[0][1] = 0

    for i in range(1, len(prices)):
        dp[i][0] = max(dp[i-1][0], -prices[i])
        dp[i][1] = max(dp[i-1][1], prices[i]+dp[i-1][0])

    print(dp)


def dp02():  # 不正确
    if len(prices) == 0:
        return 0

    """
    0-不操作
    1-买入
    2-卖出
    """
    dp = [[0]*3 for _ in range(len(prices))]

    dp[0][0] = 0
    dp[0][1] = -prices[0]
    dp[0][2] = 0

    for i in range(1, len(prices)):
        dp[i][0] = dp[i-1][0]
        dp[i][1] = max(dp[i-1][0], dp[i-1][2], -prices[i])
        dp[i][2] = max(dp[i-1][0], prices[i]+dp[i-1][1])

    print(dp)


if __name__ == '__main__':
    dp01()
    dp02()
