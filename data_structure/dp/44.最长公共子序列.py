# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/15 14:34
# @User   : RANCHODZU
# @File   : 44.最长公共子序列.py
# @e-mail : zushoujie@ghgame.cn
text1 = "abcde"
text2 = "ace"
def dp01():
    # dp[i][j]:text1的前i个和text2的前j个的最长公共子序列为dp[i][j]
    dp = [[0] * (len(text2)+1) for _ in range(len(text1)+1)]
    for i in range(1, len(text1)+1):
        for j in range(1, len(text2)+1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    print(dp)
    print(dp[len(text1)][len(text2)])
    lcs = ''
    i, j = len(text1), len(text2)
    while i > 0:
        while j > 0:
            if dp[i][j] == dp[i - 1][j]:
                i -= 1
            elif dp[i][j] == dp[i][j-1]:
                j -= 1
            else:
                lcs += text1[i-1]
                i -= 1
                j -= 1

    print(lcs[::-1])


if __name__ == '__main__':
    dp01()

