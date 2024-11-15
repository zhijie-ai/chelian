# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/15 15:16
# @User   : RANCHODZU
# @File   : 45.不相交的线.py
# @e-mail : zushoujie@ghgame.cn
A = [2, 5, 1, 2, 5]
B = [10, 5, 2, 1, 5, 2]


def dp01():
    # dp[i][j]:text1的前i个和text2的前j个的最长公共子序列为dp[i][j]
    dp = [[0] * (len(B)+1) for _ in range(len(A)+1)]

    for i in range(1, len(A) + 1):
        for j in range(1, len(B) + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    print(dp)
    print(dp[len(A)][len(B)])


if __name__ == '__main__':
    dp01()
