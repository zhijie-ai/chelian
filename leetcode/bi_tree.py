# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/9/17 13:42                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
# 深度优先遍历算法(前序，中序，后序遍历)-> 递归法
# 还有一种迭代法

def preorderTraversal(root):
    if root == None:
        return []

    res = []
    stack = [root]
    while stack:
        node = stack.pop()
        res.apped(node.val)
        if node.right :
            stack.append(node.right)
        if node.left :
            stack.append(node.left)
    return res

# 层次遍历法
def levelOrder(root):
    res = []
    from collections import deque
    if root == None:
        return []

    de = deque([root])
    while de:
        size = len(de)
        r = []
        for _ in range(size):
            node = de.popleft()
            r.append(node.val)
            if node.left:
                de.append(node.left)
            if node.right:
                de.append(node.right)
        res.append(r)
    return res