# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/9/16 15:43                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
"""
对撞指针：两个指针方向相反。适合解决查找有序数组中满足某些约束条件的一组元素问题、字符串反转问题。
快慢指针：两个指针方向相同。适合解决数组中的移动、删除元素问题，或者链表中的判断是否有环、长度问题。
分离双指针：两个指针分别属于不同的数组 / 链表。适合解决有序数组合并，求交集、并集问题。

作者：ShowMeCoding
链接：https://www.jianshu.com/p/9f0fe8b08536
来源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
"""

# 对撞指针
def max_area(height):
    area = 0
    left = 0
    right = len(height) - 1
    while left< right:
        w = right-left
        h = min(height[left], height[right])
        tmp = w*h
        if tmp> area:
            area = tmp
        if height[left] < height[right]:
            left += 1
        else:
            right-=1
    return area

# 快慢指针
def removeDuplicates(nums):
    slow = 0
    fast = 0
    while fast < len(nums):
        if nums[slow] != nums[fast]:
            slow += 1
            nums[slow] == nums[fast]
        fast += 1
    return slow + 1

# 最接近的3数之和
def threeSumClosest(nums, target):
    nums.sort()
    lenght = len(nums)
    result = float('inf')
    for i in range(2, lenght):
        left = 0
        right = i - 1
        while left < right:
            total = nums[left] + nums[right] + nums[i]
            if abs(target- total) < abs(target-result):
                result = total
            if total < target:
                left += 1
            else:
                right -= 1
    return result

def removeElement( nums, val: int) -> int:
    left = 0
    right = 0
    # 将非 val 值的元素进行前移，left 指针左边均为处理好的非 val 值元素，
    # 而从 left 指针指向的位置开始， right 指针左边都为 val 值
    # 1 [3,2,2,3]
    # 2 [3,3,2,2]  left += 1 --> 2
    while right < len(nums):
        if nums[right] != val:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
        right += 1

    print(nums)
    return left

# 数组中的最长山脉问题
def longestMountain(arr):
    result = 0
    l = len(arr)
    for i in range(1, l-1):
        if arr[i] > arr[i-1] and arr[i]< arr[i+1]:
            left = i-1
            right = i + 1
            while arr[left] > arr[left - 1] and left > 0:
                left -= 1
            while arr[right] > arr[right + 1] and right < l - 1:
                right += 1
            result = max(right - left + 1, result)
    return result

def increasingTriplet(nums) -> bool:
    # 先定义两个极大值
    a = float('inf')
    b = float('inf')
    # 然后寻找连续三个递增的数组
    for num in nums:
        # 使a尽可能小
        if num <= a:
            a = num
        # 使b尽可能小
        elif num <= b:
            b = num
        # 在同时满足上面两个条件的情况下
        else:
            return True
    return False

# 回文子串
# https://www.programmercarl.com/0647.%E5%9B%9E%E6%96%87%E5%AD%90%E4%B8%B2.html#%E5%85%B6%E4%BB%96%E8%AF%AD%E8%A8%80%E7%89%88%E6%9C%AC
def countSubstrings(s):
    def extend(s, i, j):
        count = 0
        while i>=0 and j< len(s) and s[i] == s[j]:
            count+=1
            i -=1
            j += 1
        return count
    for k in range(len(s)):
        cnt = 0
        cnt += extend(s, k, k)# 以单个字符为中心
        cnt += extend(s, k, k+1)# 以2个字符为中心
    return cnt

# 合并2个有序数组
def merge(num1, num2):
    num = [0] * (len(num1)+len(num2))
    p1 = 0
    p2 = 0
    ind = 0
    while p1 < len(num1) and p2 < len(num2):
        if num1[p1] < num2[p2]:
            num[ind] = num1[p1]
            p1 += 1
            ind +=1
        else:
            num[ind] = num2[p2]
            p2 += 1
            ind +=1

    if p1 == len(num1):
        # for i in range(p2, len(num2)):
        #     num[ind] = num2[i]
        #     ind+=1
        num[ind:] = num2[p2:]

    if p2 == len(num2):
        for i in range(p1, len(num1)):
            num[ind] = num1[i]
            ind +=1

    return num


def threeSum( nums ):
    if len(nums) < 3: return []
    nums, res = sorted(nums), []
    for i in range(len(nums) - 2):
        cur, l, r = nums[i], i + 1, len(nums) - 1
        if res != [] and res[-1][0] == cur: continue # Drop duplicates for the first time.

        while l < r:
            if cur + nums[l] + nums[r] == 0:
                res.append([cur, nums[l], nums[r]])
                # Drop duplicates for the second time in interation of l & r. Only used when target situation occurs, because that is the reason for dropping duplicates.
                while l < r - 1 and nums[l] == nums[l + 1]:
                    l += 1
                while r > l + 1 and nums[r] == nums[r - 1]:
                    r -= 1
            if cur + nums[l] + nums[r] > 0:
                r -= 1
            else:
                l += 1
    return res

def reverseStr( s, k) -> str:
    """
    1. 使用range(start, end, step)来确定需要调换的初始位置
    2. 对于字符串s = 'abc'，如果使用s[0:999] ===> 'abc'。字符串末尾如果超过最大长度，则会返回至字符串最后一个值，这个特性可以避免一些边界条件的处理。
    3. 用切片整体替换，而不是一个个替换.
    """
    def reverse_substring(text):
        left, right = 0, len(text) - 1
        while left < right:
            text[left], text[right] = text[right], text[left]
            left += 1
            right -= 1
        return text

    res = list(s)

    for cur in range(0, len(s), 2 * k):
        res[cur: cur + k] = reverse_substring(res[cur: cur + k])

    return ''.join(res)


if __name__ == '__main__':
    li = [1,8,6,2,5,4,8,3,7]
    # print(max_area(li))
    # print(removeDuplicates(li))
    # nums = [0,1,2,2,3,0,4,2]
    # val = 2
    # print(removeElement(nums, val))
    # arr = [2,1,4,7,3,2,5]
    # print(longestMountain(arr))
    # nums = [2,1,5,0,4,6]
    # print(increasingTriplet(nums))
    num1 = [1,2,3, 4]
    num2 = [2,5,6]
    # print(merge(num1,num2))
    num1 = [1,2,-3, 4]
    # print(threeSum(num1))
    s = "abcdefghijklmno"
    k = 3
    print(reverseStr(s, k))