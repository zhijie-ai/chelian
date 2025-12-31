# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2025/12/31 16:58                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
def dp01():
    coins = [1, 2, 5]
    amount = 5
    dp = [0] * (amount + 1)
    dp[0] = 1
    for i in range(len(coins)):
        # 遍历背包
        for j in range(coins[i], amount + 1):
            dp[j] += dp[j - coins[i]]
        print(dp)


def dp02():
    coins = [1, 2, 5]
    amount = 5
    dp = [0] * (amount + 1)
    dp[0] = 1


        # 遍历背包
    for j in range(0, amount + 1):
        for i in range(len(coins)):
            if (j >= coins[i]):
                dp[j] += dp[j - coins[i]]

        print(dp)


if __name__ == '__main__':
    dp01()
    print('*'*20)
    dp02()