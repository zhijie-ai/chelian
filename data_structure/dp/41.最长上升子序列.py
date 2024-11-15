# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/14 17:32
# @User   : RANCHODZU
# @File   : 41.最长上升子序列.py
# @e-mail : zushoujie@ghgame.cn

a = [0, 1, 0, 3, 2]
a =  [10,9,2,5,3,7,101,18]
def dp01(s):
    n = len(s)
    if n <= 1:
        return n

    dp = [1] * n

    result = 1
    for i in range(1, n):
        for j in range(i):
            if s[i] > s[j]:
                dp[i] = max(dp[i], dp[j]+1)

        result = max(result, dp[i])
    print(dp)
    print(result)


if __name__ == '__main__':
    dp01(a)

