# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/14 16:03
# @User   : RANCHODZU
# @File   : 30.打家劫舍II.py
# @e-mail : zushoujie@ghgame.cn

rob = [2, 7, 9, 3, 1]
def dp01(nums):
    def rob_range(nums, start, end):
        if end == start:
            return nums[start]
        prev_max = nums[start]
        curr_max = max(nums[start], nums[start+1])

        for i in range(start+2, end+1):
            temp = curr_max
            curr_max = max(prev_max+nums[i], curr_max)
            prev_max = temp

        return curr_max

    n = len(nums)
    if n == 0:
        return 0
    if n == 1:
        return nums[0]

    result1 = rob_range(nums, 0, len(nums) - 2)  # 情况二
    result2 = rob_range(nums, 1, len(nums) - 1)  # 情况二

    print(max(result1, result2))
    return max(result1, result2)


if __name__ == '__main__':
    dp01(rob)