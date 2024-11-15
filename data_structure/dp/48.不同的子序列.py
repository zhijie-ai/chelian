# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/15 17:48
# @User   : RANCHODZU
# @File   : 48.不同的子序列.py
# @e-mail : zushoujie@ghgame.cn
s = 'rabbbit'
s = 'baegg'
t = 'rabbit'
t = 'bag'

def dp01():
    # dp[i][j]：s的前i个字符的字符串中出现t的前j个字符的字符串次数
    dp = [[0] * (len(t)+1) for _ in range(len(s)+1)]

    for i in range(len(s)):
        dp[i][0] = 1
    for j in range(1, len(t)):
        dp[0][j] = 0

    for i in range(1, len(s)+1):
        for j in range(1, len(t)+1):
            if s[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1] + dp[i-1][j]

            else:
                dp[i][j] = dp[i-1][j]

    print(dp[1:])
    print(dp[-1][-1])


if __name__ == '__main__':
    dp01()