# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/7 17:58
# @User   : RANCHODZU
# @File   : 24.完全平方数.py
# @e-mail : zushoujie@ghgame.cn
import math
n = 12


def dp1d():
    dp = [float('inf')] * (n+1)
    dp[0] = 0
    for i in range(1, int(math.sqrt(n))+1):
        for j in range(n+1):
            if j >= i*i:
                dp[j] = min(dp[j], dp[j-i*i]+1)
    print(dp)

def dp2d():
    # dp[i][j] 表示从[0,i]任选且和为j完全平方数的最少数量
    dp = [[float('inf')] * (n+1) for _ in range(int(math.sqrt(n))+1)]
    dp[0][0] = 0
    print(int(math.sqrt(n)))

    for i in range(1, int(math.sqrt(n))+1):
        for j in range(n+1):
            if j >= i*i:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - i*i] + 1)
            else:
                dp[i][j] = dp[i - 1][j]

    print(dp[1:])


if __name__ == '__main__':
    dp1d()
    dp2d()