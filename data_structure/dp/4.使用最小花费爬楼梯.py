# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/20 15:56
# @User   : RANCHODZU
# @File   : 4.使用最小花费爬楼梯.py
# @e-mail : zushoujie@ghgame.cn

cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]

def dp01():
    dp = [0] * (len(cost)+1)

    dp[0] = 0  # 初始值，表示从起点开始不需要花费体力
    dp[1] = 0  # 初始值，表示经过第一步不需要花费体力

    for i in range(2, len(cost) + 1):
        # 在第i步，可以选择从前一步（i-1）花费体力到达当前步，或者从前两步（i-2）花费体力到达当前步
        # 选择其中花费体力较小的路径，加上当前步的花费，更新dp数组
        dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])

    print(dp)
    return dp[len(cost)]  # 返回到达楼顶的最小花费


if __name__ == '__main__':
    dp01()