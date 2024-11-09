# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/9/14 11:19                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------

# 两数相加
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# class Solution:
#     def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
#         head_node = ListNode(0)
#         node = head_node
#         carry = 0
#         while l1 or l2:
#             l1_num = l1.val if l1 else 0
#             l2_num = l2.val if l2 else 0
#             num = (l1_num+l2_num+carry) % 10
#             carry = 1 if (l1_num+l2_num+carry)>=10 else 0
#             node.next = ListNode(num)
#             node = node.next
#             l1 = l1.next if l1 else None
#             l2 = l2.next if l2 else None
#         if carry == 1:
#             node.next = ListNode(1)
#         head_node = head_node.next
#         return head_node

def reverse(x: int) -> int:
    s = str(x)
    if s[0] =='-':
        t = list(s[1:][::-1])
        t.insert(0, '-')
        s = ''.join(t)
    else:
        s = s[::-1]
    x = int(s)

    if x>= (2**31)-1 or x< -2**31:
        return 0
    return x

# 最接近的3个数之和
import numpy as np
def threeSumClosest(nums: list, target: int) -> int:
    nums.sort()
    ret = float('inf')
    length = len(nums)
    for i in range(length-2):
        left = i +1
        right = length-1
        while left < right:
            tmp = nums[i] + nums[left] + nums[right]
            ret = tmp if abs(tmp-target)<abs(ret-target) else ret
            if tmp == target:
                return tmp
            if tmp> target:
                right -= 1
            else:
                left+=1
    return ret



if __name__ == '__main__':
    x = -123
    # print(reverse(x))
    nums = [-1, 2, 1,-4]
    target = 1
    print(threeSumClosest(nums, target))
