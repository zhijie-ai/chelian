# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/20 17:59
# @User   : RANCHODZU
# @File   : 翻转字符串里的单词.py
# @e-mail : zushoujie@ghgame.cn

def removeExtraSpaces(s):
    slow = 0
    for i in range(len(s)):
        if s[i] != ' ':  # //遇到非空格就处理，即删除所有空格。
            if slow != 0:
                s[slow] = ' '  # //手动控制空格，给单词之间添加空格。slow != 0说明不是第一个单词，需要在单词前添加空格。
                slow += 1
            while (i < len(s) and s[i] != ' '):
                s[slow] = s[i]
                slow += 1
                i += 1

    print(s)


if __name__ == '__main__':
    st = "  hello world!  "
    removeExtraSpaces(st)
