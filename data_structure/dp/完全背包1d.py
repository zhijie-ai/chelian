# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/5 18:21
# @User   : RANCHODZU
# @File   : 完全背包1d.py
# @e-mail : zushoujie@ghgame.cn
# 完全背包先物品再背包和先背包再物品是没区别的
def dp01():
    weight = [1, 3, 4]
    value = [15, 20, 70]
    bagWeight = 5
    dp = [0] * (bagWeight + 1)
    for i in range(len(weight)):  # 遍历物品
        for j in range(weight[i], bagWeight + 1):  # 遍历背包容量
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
    print(dp)


def dp02():
    weight = [1, 3, 4]
    value = [15, 20, 70]
    bagWeight = 5

    dp = [0] * (bagWeight + 1)

    for j in range(bagWeight + 1):  # 遍历背包容量
        for i in range(len(weight)):  # 遍历物品
            if j - weight[i] >= 0:
                dp[j] = max(dp[j], dp[j - weight[i]] + value[i])

    print(dp)


# https://programmercarl.com/0377.%E7%BB%84%E5%90%88%E6%80%BB%E5%92%8C%E2%85%A3.html#%E5%85%B6%E4%BB%96%E8%AF%AD%E8%A8%80%E7%89%88%E6%9C%AC
# 看似求组合，实际上是排列。
def dp03():
    nums = [1, 2, 3]
    target = 4
    dp = [0] * (target + 1)  # 创建动态规划数组，用于存储组合总数
    dp[0] = 1  # 初始化背包容量为0时的组合总数为1

    for i in range(1, target + 1):  # 遍历背包容量
        for j in nums:  # 遍历物品列表
            if i >= j:  # 当背包容量大于等于当前物品重量时
                dp[i] += dp[i - j]  # 更新组合总数

    print(dp)
    return dp[-1]  # 返回背包容量为target时的组合总数


def compare():
    # 修改数据：让物品不是按重量排序，且价值比例更复杂
    weight = [3, 1, 4]  # 注意：1不在最前面了
    value = [20, 15, 30]
    bagWeight = 4

    # 方法1：先物品后背包
    dp1 = [0] * (bagWeight + 1)
    for i in range(len(weight)):
        for j in range(weight[i], bagWeight + 1):
            dp1[j] = max(dp1[j], dp1[j - weight[i]] + value[i])
    print("先物品后背包:", dp1)  # [0, 15, 30, 45, 60]

    # 方法2：先背包后物品
    dp2 = [0] * (bagWeight + 1)
    for j in range(bagWeight + 1):
        for i in range(len(weight)):
            if j >= weight[i]:
                dp2[j] = max(dp2[j], dp2[j - weight[i]] + value[i])
    print("先背包后物品:", dp2)  # [0, 15, 30, 45, 60]

    # 看起来还是相同？再换一个

def extreme_example():
    weight = [2, 3, 6]  # 更稀疏的重量
    value = [5, 9, 20]  # 价值比例：2.5, 3.0, 3.33
    bagWeight = 7

    # 方法1：先物品后背包
    dp1 = [0] * (bagWeight + 1)
    for i in range(len(weight)):
        for j in range(weight[i], bagWeight + 1):
            dp1[j] = max(dp1[j], dp1[j - weight[i]] + value[i])
    print("先物品后背包:", dp1)
    print("容量7的值:", dp1[7])  # 最优解：3+2+2=14

    # 方法2：先背包后物品
    dp2 = [0] * (bagWeight + 1)
    for j in range(bagWeight + 1):
        for i in range(len(weight)):
            if j >= weight[i]:
                dp2[j] = max(dp2[j], dp2[j - weight[i]] + value[i])
    print("先背包后物品:", dp2)
    print("容量7的值:", dp2[7])


if __name__ == '__main__':
    dp01()
    dp02()
    dp03()
    compare()
    extreme_example()
