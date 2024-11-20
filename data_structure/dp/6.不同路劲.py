# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/20 16:15
# @User   : RANCHODZU
# @File   : 6.不同路劲.py
# @e-mail : zushoujie@ghgame.cn
m, n = 3, 7

def dp01():
    dp = [[0] * n for _ in range(m)]  # 表示从（0 ，0）出发，到(i, j) 有dp[i][j]条不同的路径。

    for i in range(m):
        dp[i][0] = 1
    for j in range(n):
        dp[0][j] = 1

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j]+dp[i][j-1]

    print(dp)
    return dp[-1][-1]


if __name__ == '__main__':
    dp01()