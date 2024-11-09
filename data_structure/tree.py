# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/9/15 17:20                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
from collections import deque
class Tree():
    def __init__(self,data):
        self.data=data
        self.lchild=None
        self.rchild=None

a=Tree("A")
b=Tree("B")
c=Tree("C")
d=Tree("D")
e=Tree("E")
f=Tree("F")
g=Tree("G")

e.lchild=a
e.rchild=g
a.rchild=c
c.lchild=b
c.rchild=d
g.rchild=f

# 前序遍历
def pre_read(root):
    if root:
        print(root.data,end=' ')
        pre_read(root.lchild)
        pre_read(root.rchild)

# 中序遍历
def mid_read(root):
    if root:
        mid_read(root.lchild)
        print(root.data,end=' ')
        mid_read(root.rchild)

#第三种方法：后序遍历
def back_read(root):
    if root:
        back_read(root.lchild)
        back_read(root.rchild)
        print(root.data,end=' ')



if __name__ == '__main__':
    # pre_read(e)
    # print("\n1.前序遍历")

    # mid_read(e)
    # print("\n2.中序遍历")

    print("\n3.后序遍历")
    back_read(e)
