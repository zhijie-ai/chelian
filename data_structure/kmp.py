# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/10/17 18:12
# @User   : RANCHODZU
# @File   : kmp.py
# @e-mail : zushoujie@ghgame.cn

def calc_lps(pattern):
    """
    计算部分匹配表（Longest Prefix which is also Suffix）
    :param pattern: 模式字符串
    :return: 部分匹配表（列表）
    """
    pmt = [0] * len(pattern)
    j = 0
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[j]:
            j += 1
            pmt[i] = j
            i += 1
        else:
            if j != 0:
                j = pmt[j-1]
            else:
                pmt[i] = 0
                i += 1

    return pmt

def kmp_search(text, pattern):
    """
    KMP算法主函数，用于在文本中查找模式
    :param text: 文本字符串
    :param pattern: 模式字符串
    :return: 匹配的所有起始索引列表
    """
    n = len(text)
    m = len(pattern)

    lps = calc_lps(pattern)
    i = j = 0
    indices = []
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == m:
            indices.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return indices


if __name__ == '__main__':
    # text = "ABABDABACDABABCABAB"
    # pattern = "ABABCABAB"
    #
    # matches = kmp_search(text, pattern)
    # print(f"Pattern found at indices: {matches}")

    text = open('train.tags.de-en.en', encoding='utf8').read()
    pattern = '<translator href="http://www.ted.com/profiles/258692">Geoffrey Chen</translator>'
    print(kmp_search(text, pattern))