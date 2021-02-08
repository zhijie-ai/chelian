#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/5/5 15:42                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

def merge(a,b):
    c = []
    h=j=0
    while h<len(a) and j<len(b):
        if a[h]<b[j]:
            c.append(a[h])
            h+=1
        else:
            c.append(b[j])
            j+=1

    if h==len(a):
        for i in b[j:]:
            c.append(i)
    if j == len(b):
        for i in a[h:]:
            c.append(i)

    return c

def merge_sort(lists):
    if len(lists)<=1:
        return lists

    mid  = len(lists)//2
    left =merge_sort(lists[:mid])
    right =merge_sort(lists[mid:])
    return merge(left,right)

if __name__ == '__main__':
    a = [4,7,8,3,5,9]
    print(merge_sort(a))
