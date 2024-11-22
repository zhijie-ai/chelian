# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/7 10:54
# @User   : RANCHODZU
# @File   : 22.爬楼梯（进阶版）.py
# @e-mail : zushoujie@ghgame.cn

m, n = 3, 6


def dp01():
    dp = [0] * (n + 1)
    dp[0] = 1
    ways = [[] for _ in range(n+1)]
    ways[0].append([])

    for j in range(1, n+1):
        for i in range(1, m+1):
            if j >= i:
                dp[j] += dp[j-i]
                for w in ways[j-i]:
                    ways[j].append(w+[i])
            else:
                ways[j] = ways[j][:]
    print(dp)
    print(ways[n])


# 注意初始化和遍历范围的关系。
# 当前初始化的是第一列, 那么j可以从1开始也可以从0开始, i必须从0开始
def dp02():
    dp = [[0] * (n+1) for _ in range(m)]
    for i in range(m):  # 只能通过这种方式初始化
        dp[i][0] = 1

    for j in range(1, n+1):
        for i in range(m):
            if j >= i+1:
                dp[i][j] = dp[i-1][j] + dp[-1][j-(i+1)]
            else:
                dp[i][j] = dp[i - 1][j]

    print(dp)

"""
初始化的是dp[-1][0],ij都从0开始遍历
"""
def dp02_():
    dp = [[0] * (n+1) for _ in range(m)]
    dp[-1][0] = 1

    for j in range(n+1):
        for i in range(m):
            if j >= i+1:
                dp[i][j] = dp[i-1][j] + dp[-1][j-(i+1)]
            else:
                dp[i][j] = dp[i - 1][j]

    print(dp)


def dp03():
    dp = [[0] * (n+1) for _ in range(m+1)]
    dp[0][0] = 1  # 边界条件：到达第0阶的方法只有一种，就是不动

    # for i in range(m):
    #     dp[i][0] = 1
    for j in range(n+1):
        for i in range(1, m+1):
            if j >= i:
                dp[i][j] = dp[i-1][j] + dp[-1][j-i]
            else:
                dp[i][j] = dp[i - 1][j]

    print(dp[1:])


if __name__ == '__main__':
    dp01()
    dp02()
    dp02_()
    dp03()