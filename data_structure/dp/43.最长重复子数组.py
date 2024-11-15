# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/15 11:35
# @User   : RANCHODZU
# @File   : 43.最长重复子数组.py
# @e-mail : zushoujie@ghgame.cn
A = [1, 2, 3, 2, 1]
B = [3, 2, 1, 4, 7]

def dp2d():
    dp = [[0]*(len(A)+1) for _ in range(len(B)+1)]

    result = 0
    for i in range(1, len(A)+1):
        for j in range(1, len(B)+1):
            if A[i-1] == B[j-1]:
                dp[i][j] = dp[i-1][j-1]+1

            if dp[i][j] > result:
                result = dp[i][j]

    print(dp)
    print(result)
    return result


if __name__ == '__main__':
    dp2d()

