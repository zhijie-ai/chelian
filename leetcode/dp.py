# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/9/17 15:56                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
# 回文子串的数量
def countSubstrings(s):
    dp = [[False] * len(s) for _ in range(len(s))]
    result = 0
    for i in range(len(s)-1, -1, -1):
        for j in range(i, len(s)):
            if s[i] == s[j] and (j -i<= 1 or dp[i+1][j-1]):
                result += 1
                dp[i][j] = True
    return  result

# 最长的回文子序列
def longestPalindromeSubseq(s):
    dp = [[0] * len(s) for _ in range(len(s))]
    for i in range(len(s)):
        dp[i][i] =1

    for i in range(len(s))[::-1]:
        for j in range(i+1, len(s)):
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1]+2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    return dp[0][-1]

# 最长公共子序列
def longestCommonSubsequence(s1, s2):
    len1, len2 = len(s1)+1, len(s2)+1
    dp = [[0 for _ in range(len1)] for _ in range(len2)] # 先对dp数组做初始化操作
    for i in range(1, len1):
        for j in range(1, len2):
            if s1[i] == s2[j]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[-1][-1]

# 01背包问题
def _01_wei_bag_problem1(bag_size, weight, value):
    rows, cols = len(weight), bag_size + 1
    dp = [[0]* cols for _ in range(rows)]
    # 初始化dp
    for i in range(rows):
        dp[i][0] = 0
    first_item_weight, first_item_value = weight[0], value[0]
    for j in range(1, cols):
        if first_item_weight <= j:
            dp[0][j] = first_item_value

    # 更新dp数组: 先遍历物品, 再遍历背包.
    for i in range(1, rows):
        cur_weight, cur_val = weight[i], value[i]
        for j in range(1, cols):
            if cur_weight> j:# 说明背包装不下当前物品.
                dp[i][j] = dp[i-1][j] # 所以不装当前物品.
            else:
                # 定义dp数组: dp[i][j] 前i个物品里，放进容量为j的背包，价值总和最大是多少。
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-cur_weight]+cur_val)
    return dp

# 编辑距离
def minDistance(word1, word2):
    dp = [[0]*(len(word2)+1) for _ in range(len(word1)+1)]
    for i in range(len(word1)+1):
        dp[i][0] = i
    for j in range(len(word2)+1):
        dp[0][j] = j

    for i in range(1, len(dp)):
        for j in range(1, len(dp[0])):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1] +1, dp[i-1][j-1]+1)
    return dp[-1][-1]


if __name__ == '__main__':
    # bag_size = 4
    # weight = [1, 3, 4]
    # value = [15, 20, 30]
    # print(_01_wei_bag_problem1(bag_size, weight, value))
    word1 = "horse1111"
    word2 = "ros22222"
    print(minDistance(word1, word2))
