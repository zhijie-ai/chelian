# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/1 11:29
# @User   : RANCHODZU
# @File   : __init__.py.py
# @e-mail : zushoujie@ghgame.cn

#二维数组情况，不管是01背包还是完全背包，只要保证i是正序就可以了，也就是说先物品先背包都可以，内层正序或者逆序均行
# 一维数组情况，01背包，先物品后背包，且背包逆序(01背包问题中，背包只能逆序，如果先背包后物品，前面的状态还没更新呢)，
#   纯完全背包，先物品或者先背包都可以，内层正序.纯完全背包是凑成背包最大价值是多少
#   如果问装满背包有几种方式的话？ 那么两个for循环的先后顺序就有很大区别了
# 纯完全背包问题求装满背包的最大价值是多少，和凑成总和的元素有没有顺序没关系，即：有顺序也行，没有顺序也行！
"""
在求装满背包有几种方案的时候，认清遍历顺序是非常关键的。
如果求组合数就是外层for循环遍历物品，内层for遍历背包。
如果求排列数就是外层for遍历背包，内层for循环遍历物品。
当j=6时，既可以从dp[5]加1得到（1+...），也可以从dp[1]加5得到（5+...）因此包含了{1,5}和{5,1}两种排列

j = 6 ⭐最关键步骤
    遍历物品i：
        i=0（coin=1）：dp[6] += dp[5] = 0+2 = 2
            含义：最后一步用1元硬币，前面要凑成5元
            已知dp[5]=2对应：
                {1,1,1,1,1} → 加上1元 → {1,1,1,1,1,1}
                {5} → 加上1元 → {5,1}
            所以添加2种新方法
        i=1（coin=5）：dp[6] += dp[1] = 2+1 = 3
            含义：最后一步用5元硬币，前面要凑成1元
            已知dp[1]=1对应{1}
            新方法：{1,5}
            与上面重复？不重复！因为这是不同的顺序：
                之前通过dp[5]+1得到的是{5,1}（5在前，1在后）
                现在通过dp[1]+5得到的是{1,5}（1在前，5在后）

从dp[5]加1元硬币：最后一步是1元
从dp[1]加5元硬币：最后一步是5元
"""

"""
背包问题是一个资源消耗问题，而股票问题是一个状态庄毅+资金流动问题。
背包通常优化物品选择，股票优化买卖时机
股票问题的本质是在时间序列上操作一个有限状态机
dp[天数][交易次数][持有状态]
k最多为1时，之所以不用3维来表示，是因为持有：dp[i][0][1] = -∞ → 不可能出现（没有交易机会，不能买入，自然无法持有）
dp[0][1][0] = 0 → 第 0 天，有交易机会，不持有股票，利润 0
dp[0][1][1] = -prices[0] → 第 0 天，有交易机会，买入股票，利润为负的股价
dp[0][0][0] = 0 → 第 0 天，无交易机会，不持有，利润 0
dp[0][0][1] = -∞ → 不可能状态
总结：k=1或∞时，只需要2维来表述就可以了，其他的必须用3维来表示
其实可以用 state = k_idx*2 + s 来压缩二三维，但一般很少这么勇，无效状态冗余且易干扰，转移关系不直观，复杂且易出错
一般用二维 DP 显式表示(去掉i维)或者状态机视角

资源调度问题，能源管理问题，库存管理问题
# 通用框架
def generic_dp():
    state = (i, capacity)  # 考虑前i个物品，剩余容量
    decision = 0 or 1      # 不选或选
    next_state = (i+1, capacity - weight*decision)
    reward = value * decision

# 特殊化为背包问题
def knapsack_dp():
    # 去掉时间维度（物品顺序无关）
    state = capacity
    decision = 0 or 1
    next_state = capacity - weight*decision
    reward = value * decision
    
如果将k定义为已完成的交易次数，则卖出会消耗次数
如果将k定义为剩余交易次数或者已用掉交易次数，则买入会消耗次数
已用掉：买入就算"用掉"一次交易机会
已完成：卖出才算"完成"一次完整交易
如果用状态机的方式，注意更新顺序
def maxProfit_k2_best(prices):
    if not prices:
        return 0
    
    # 四个明确的状态
    # after_first_buy: 第一次买入后的最大利润
    # after_first_sell: 第一次卖出后的最大利润
    # after_second_buy: 第二次买入后的最大利润
    # after_second_sell: 第二次卖出后的最大利润
    
    after_first_buy = -prices[0]
    after_first_sell = 0
    after_second_buy = -prices[0]
    after_second_sell = 0
    
    for price in prices[1:]:
        # 更新顺序重要：先更新卖出，再更新买入
        after_second_sell = max(after_second_sell, after_second_buy + price)
        after_second_buy = max(after_second_buy, after_first_sell - price)
        after_first_sell = max(after_first_sell, after_first_buy + price)
        after_first_buy = max(after_first_buy, -price)
    
    return after_second_sell
    
三维DP
二维DP
状态机变量法
通用状态机（通用DP）：hold = [-float('inf')] * (k + 1) cash = [0] * (k + 1)
"""