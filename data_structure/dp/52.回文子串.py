# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/18 18:25
# @User   : RANCHODZU
# @File   : 52.回文子串.py
# @e-mail : zushoujie@ghgame.cn


# 动态规划版本
def dp01(s):
    """
    当s[i]与s[j]相等时，这就复杂一些了，有如下三种情况

    情况一：下标i 与 j相同，同一个字符例如a，当然是回文子串
    情况二：下标i 与 j相差为1，例如aa，也是回文子串
    情况三：下标：i 与 j相差大于1的时候，例如cabac，此时s[i]与s[j]已经相同了，
    我们看i到j区间是不是回文子串就看aba是不是回文就可以了，那么aba的区间就是 i+1 与 j-1区间，
    这个区间是不是回文就看dp[i + 1][j - 1]是否为true。
    :param s:
    :return:
    """
    dp = [[False] * len(s) for _ in range(len(s))]

    result = 0

    for i in range(len(s)-1, -1, -1):  # 注意遍历顺序
        for j in range(i, len(s)):
            if s[i] == s[j]:
                if j-i <= 1:  # 情况一和情况二
                    result += 1
                    dp[i][j] = True
                elif dp[i+1][j-1]:  # 情况三
                    result += 1
                    dp[i][j] = True

    print(dp)
    print(result)
    return result


# 动态规划简洁版
def dp02(s):
    dp = [[False] * len(s) for _ in range(len(s))]
    result = 0
    for i in range(len(s) - 1, -1, -1):  # 注意遍历顺序
        for j in range(i, len(s)):
            if s[i] == s[j] and (j - i <= 1 or dp[i + 1][j - 1]):
                result += 1
                dp[i][j] = True

    print(dp)
    return result


# 双指针法
def dp03(s):
    def extend(s, i, j, n):
        res = 0
        while i >= 0 and j < n and s[i] == s[j]:
            i -= 1
            j += 1
            res += 1

        return res

    result = 0
    for i in range(len(s)):
        result += extend(s, i, i, len(s))  # 以i为中心
        result += extend(s, i, i+1, len(s))  # 以i和i+1为中心

    print(result)
    return result


if __name__ == '__main__':
    st = 'abcabc'
    dp01(st)
    dp02(st)
    dp03(st)
