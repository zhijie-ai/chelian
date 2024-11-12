# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/6 14:42
# @User   : RANCHODZU
# @File   : 完全背包2d.py
# @e-mail : zushoujie@ghgame.cn

"""
当都使用前i个的定义时，组合问题和排列问题的初始化定义不一样
完全背包有2个问题，一个是max问题，凑成背包最大价值是多少，一个是组合数的问题
dp01是组合问题，dp02是排列问题，遍历顺序不同
2,3,4都是解决的排列问题，不一样的地方在于，d[i][j]的定义，3中的定义更接近于原始的二维数组的定义，即[0,i]索引中任选凑成j排列个数
2的定义则是，索引为[0,i]的数分别作为排列的最后一个元素，且排列之和为j的所有排列的总数
3的定义则是，[0,i]索引中任选凑成和为j的排列个数
4的定义则是，前i个数分别作为排列的最后一个元素，且排列之和为j的所有排列的总数
由于2种定义的区别，造成了dp数据有区别。比如dp[1][4],其中2为5,3和4则为6
2,4是从所有元素中选，而3则是从[0,i]的索引中选，2,4每次都用到了所有元素，3只用到了前几的元素,所以3肯定比2,4少
"""
"""
j为1时： i==0，表示把nums[0]放置在该集合的最后一个元素的位置 那么所得集合为{dp[j - nums[i]]所表示集合, 1} ，即{ dp[0]所表示集合，1}，其中dp[0]所表示集合不存在，因此所得集合为{ 1 }
j为2时：
	i==0：将1放置在集合中的最后一个位置上。所得集合为{dp[j - nums[i]]所表示集合, 1}，即{ dp[1]所表示集合，1}，因此所得集合为{ 1, 1 }
	同理，i == 1时，nums[1] == 2（即将2放在集合中的最后一个位置上），所得集合为{ dp[0]所表示集合，2}，因此所得集合为{ 2 }
	因此，j==2时的集合有{1, 1}，{ 2 }
j为3时：
	i==0，所得集合为{dp[2]所表示集合，1}，即{1, 1, 1}和{2, 1}
	i==1，所得集合为{dp[1]所表示集合, 2} ，即{1, 2}
	i==2，所得集合为{dp[0]所表示集合, 3}， 即{ 3 }
可以发现，上述出现的集合中出现了{1, 2}与{2, 1}，这就是为什么先背包后物品可以得出排列数
"""

# 零钱兑换II，原本是用java写的，改成了python， 求组合数，和下面的组合总数题目不一样
def dp01():
    coins = [1, 2, 5]
    amount = 5

    dp = [[0] * (amount + 1) for _ in range(len(coins))]

    for i in range(len(coins)):
        dp[i][0] = 1

    for j in range(coins[0], amount+1):
        dp[0][j] += dp[0][j-coins[0]]

    for i in range(1, len(coins)):
        for j in range(1, amount+1):
            if j < coins[i]:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = dp[i][j - coins[i]] + dp[i - 1][j]
    print(dp)  # [1, 1, 2, 2, 3, 4]]

# 零钱兑换
def dp01_():
    coins = [1, 2, 5]
    amount = 5

    dp = [[0] * (amount + 1) for _ in range(len(coins)+1)]
    for i in range(len(coins)+1):
        dp[i][0] = 1

    for i in range(1, len(coins)+1):
        for j in range(1, amount+1):
            if j < coins[i-1]:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = dp[i][j - coins[i-1]] + dp[i - 1][j]
    print(dp[1:])


# 前n个的定义，且只初始化一个位置
def dp01__():
    coins = [1, 2, 5]
    amount = 5

    dp = [[0] * (amount + 1) for _ in range(len(coins)+1)]
    dp[0][0] = 1
    ways = [[] for _ in range(amount + 1)]
    ways[0].append([])

    for i in range(1, len(coins)+1):
        coin = coins[i-1]
        for j in range(amount+1):
            if j >= coins[i - 1]:
                dp[i][j] = dp[i - 1][j] + dp[i][j-coin]
                for way in ways[j - coin]:
                    ways[j].append(way + [coin])
            else:
                dp[i][j] = dp[i-1][j]
                ways[j] = ways[j][:]
    print(dp[1:])
    print(ways[amount])


# 组合总和IV，看着是组合，实际上算的是排序数,blog中的python二维数组的解决方式
# https://programmercarl.com/0377.%E7%BB%84%E5%90%88%E6%80%BB%E5%92%8C%E2%85%A3.html#%E5%85%B6%E4%BB%96%E8%AF%AD%E8%A8%80%E7%89%88%E6%9C%AC
def dp02():
    nums = [1, 2, 3]
    target = 4
    dp = [[0] * (target + 1) for _ in nums]

    for i in range(len(nums)):
        dp[i][0] = 1

    # 这里不能初始化dp[0][j]。dp[0][j]的值依赖于dp[-1][j-nums[0]]
    for j in range(1, target + 1):
        for i in range(len(nums)):

            if j - nums[i] >= 0:
                dp[i][j] = (
                    # 不放nums[i]
                    # i = 0 时，dp[-1][j]恰好为0，所以没有特殊处理
                        dp[i - 1][j] +
                        # 放nums[i]。对于和为j的组合，只有试过全部物品，才能知道有几种组合方式。所以取最后一个物品dp[-1][j-nums[i]]
                        dp[-1][j - nums[i]]
                )
            else:
                dp[i][j] = dp[i - 1][j]
    print(dp)
    return dp[-1][-1]


# 借鉴的是评论中的java代码
def dp03():
    nums = [1, 2, 3]
    target = 4
    dp = [[0] * (target + 1) for _ in nums]

    for i in range(len(nums)):
        dp[i][0] = 1

    for j in range(target+1):
        if j % nums[0] == 0:
            dp[0][j] = 1

    for i in range(1, len(nums)):
        for j in range(1, target+1):
            if nums[i] > j:
                dp[i][j] = dp[i-1][j]
            else:
                for k in range(i, -1, -1):
                    if j >= nums[k]:
                        dp[i][j] += dp[i][j-nums[k]]
    print(dp)

# dp04参考的是LeetCode上的代码，和02的结果是一样的
# https://leetcode.cn/problems/combination-sum-iv/solutions/2663854/zu-he-zong-he-ivshu-xue-tui-dao-xian-gou-ap8y/
def dp04():
    nums = [1, 2, 3]
    target = 4
    dp = [[0] * (target + 1) for _ in range(len(nums)+1)]
    dp[0][0] = 1  # 配合i从1开始遍历，就是为了设置dp[:][0]=1
    ways = [[] for _ in range(target+1)]
    ways[0].append([])
    for j in range(target + 1):
        for i in range(1, len(nums) + 1):
            if j < nums[i-1]:
                dp[i][j] = dp[i-1][j]
                ways[j] = ways[j][:]
            else:
                dp[i][j] = dp[i-1][j] + dp[len(nums)][j-nums[i-1]]
                for way in ways[j-nums[i-1]]:
                    ways[j].append(way + [nums[i-1]])
    print(dp[1:])
    print(ways)


if __name__ == '__main__':
    dp01()
    dp01_()
    dp01__()
    dp02()
    dp03()
    dp04()
