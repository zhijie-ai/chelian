# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/15 17:08
# @User   : RANCHODZU
# @File   : 47.判断子序列.py
# @e-mail : zushoujie@ghgame.cn
s = "abc"
t = "ahbgdc"


def dp01():
    # dp[i][j]：dp[i][j]:s的前i个和t的前j个的相同子序列为dp[i][j]
    dp = [[0] * (len(t)+1) for _ in range(len(s)+1)]

    for i in range(1, len(s)+1):
        for j in range(1, len(t)+1):
            if s[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = dp[i][j-1]
    print(dp[1:])
    if dp[-1][-1] == len(s):
        return True


if __name__ == '__main__':
    dp01()
