# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/18 16:36
# @User   : RANCHODZU
# @File   : 50.编辑距离.py
# @e-mail : zushoujie@ghgame.cn
word1 = "horse"
word2 = "ros"


def dp01():

    # dp[i][j]:表示word1的前i个字符的字符串变成word2的前j个字符的字符串，所需要的最小的编辑距离为dp[i][j]
    dp = [[0]*(len(word2)+1) for _ in range(len(word1)+1)]

    for i in range(len(word1)+1):
        dp[i][0] = i

    for j in range(len(word2)+1):
        dp[0][j] = j

    for i in range(len(word1)+1):
        for j in range(len(word2)+1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1

    print(dp)
    return dp[-1][-1]


if __name__ == '__main__':
    dp01()
