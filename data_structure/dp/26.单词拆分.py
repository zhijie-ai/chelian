# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/8 14:59
# @User   : RANCHODZU
# @File   : 26.单词拆分.py
# @e-mail : zushoujie@ghgame.cn
s = "applepenapple"
wordDict = ["apple", "pen"]


def dp1d():
    dp = [0] * (len(s)+1)  # dp[j]表示长度为j的字符串可以拆分成一个或者多个在wordDict出现的单词
    dp[0] = 1

    for j in range(1, len(s)+1):
        for word in wordDict:
            if j >= len(word):
                dp[j] = dp[j] or (dp[j-len(word)] and word == s[j-len(word):j])
    print([int(i) for i in dp])
    return dp[len(s)]


def dp2d():
    # dp[i][j]表示前i个单词能组合成长度为j的单词
    dp = [[0] * (len(s) + 1) for _ in range(len(wordDict)+1)]
    dp[0][0] = 1

    for j in range(len(s)+1):  # 背包
        for i in range(1, len(wordDict)+1):  # 物品
            word = wordDict[i-1]
            if j >= len(word):
                dp[i][j] = dp[i-1][j] or (dp[-1][j - len(word)] and word == s[j - len(word):j])
            else:
                dp[i][j] = dp[i - 1][j]

    dp = [[int(i) for i in j] for j in dp]
    print('前n个：', dp[1:])

def dp2d_():
    # dp[i][j]表示索引为[0,i]的单词任取能组合成长度为j的单词
    dp = [[0] * (len(s) + 1) for _ in range(len(wordDict))]
    for i in range(len(wordDict)):
        dp[i][0] = 1

    for j in range(1, len(s)+1):  # 背包
        for i in range(len(wordDict)):  # 物品
            word = wordDict[i]
            if j >= len(word):
                dp[i][j] = dp[i - 1][j] or (dp[-1][j - len(word)] and word == s[j - len(word):j])
            else:
                dp[i][j] = dp[i - 1][j]

    dp = [[int(i) for i in j] for j in dp]
    print('[0,i]：', dp)


if __name__ == '__main__':
    dp1d()
    dp2d()
    dp2d_()