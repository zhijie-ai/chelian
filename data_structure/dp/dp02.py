# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/1 17:32
# @User   : RANCHODZU
# @File   : dp02.py
# @e-mail : zushoujie@ghgame.cn
from typing import List


# 分割等和子集
def can_partition(nums: List[int]) -> bool:
    if sum(nums) % 2 != 0:
        return False
    target = sum(nums) // 2
    dp = [0] * (target + 1)
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = max(dp[j], dp[j - num] + num)
    return dp


if __name__ == '__main__':
    a = [1, 5, 11, 5]
    print(can_partition(a))

