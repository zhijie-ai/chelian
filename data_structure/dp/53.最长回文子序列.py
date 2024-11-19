# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/19 09:56
# @User   : RANCHODZU
# @File   : 53.最长回文子序列.py
# @e-mail : zushoujie@ghgame.cn

def dp01(s):

    # 字符串s在[i, j]范围内最长的回文子序列的长度为dp[i][j]
    dp = [[0] * len(s) for _ in range(len(s))]

    for i in range(len(s)):
        dp[i][i] = 1

    for i in range(len(s)-1, -1, -1):
        for j in range(i+1, len(s)):
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

    print(dp)
    print(dp[0][-1])
    return dp[0][-1]


if __name__ == '__main__':
    st = 'abcabc'
    dp01(st)