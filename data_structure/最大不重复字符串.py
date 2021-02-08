#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/5/2 22:41                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

def substring(s):
    if not s:return 0
    left = 0
    max_len = 0
    cur_len = 0
    lookup = set()
    for i in s:
        cur_len +=1
        while i in lookup:
            lookup.remove(s[left])
            left+=1
            cur_len-=1
        if cur_len >max_len:max_len=cur_len
        lookup.add(i)
    return max_len

print(substring('abcdabab'))