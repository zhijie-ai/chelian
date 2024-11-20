# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/20 17:29
# @User   : RANCHODZU
# @File   : 3.替换数字.py
# @e-mail : zushoujie@ghgame.cn

def fun01(s):
    lst = list(s)  # Python里面的string也是不可改的，所以也是需要额外空间的。空间复杂度：O(n)。
    for i in range(len(lst)):
        if lst[i].isdigit():
            lst[i] = "number"
    return ''.join(lst)