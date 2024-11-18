# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/15 18:55
# @User   : RANCHODZU
# @File   : 49.两个字符串的删除操作.py
# @e-mail : zushoujie@ghgame.cn

word1 = 'sea'
word2 = 'eat'
def dp01():
    # dp[i][j]: 前i个字符串的word1和前j个字符串的word2，想要达到相等，所需要删除元素的最小次数
    dp = [[0] * (len(word2)+1) for _ in range(len(word1)+1)]

    for i in range(len(word1)+1):
        dp[i][0] = i

    for j in range(len(word2)+1):
        dp[0][j] = j

    for i in range(1, len(word1)+1):
        for j in range(1, len(word2)+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1]+2, dp[i-1][j]+1, dp[i][j-1]+1)

    print(dp)
    print(dp[-1][-1])
    return dp[-1][-1]


def dp02():  # 通过求两个字符串的最长公共子串来求解
    dp = [[0] * (len(word2)+1) for _ in range(len(word1)+1)]

    for i in range(1, len(word1)+1):
        for j in range(1, len(word2)+1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 删去最长公共子序列以外元素
    print(len(word2)+len(word1) - 2*dp[-1][-1])
    return len(word2)+len(word1) - 2*dp[-1][-1]


if __name__ == '__main__':
    dp01()
    dp02()

