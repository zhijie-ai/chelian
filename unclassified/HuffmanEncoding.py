#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/8/17 16:18
 =================知行合一=============
'''
#Huffman Encoding

#Tree-Node Type
class Node:
    def __init__(self,freq,char):
        self.left = None
        self.right = None
        self.father = None
        self.freq = freq
        self.char = char
    def isLeft(self):
        return self.father.left == self

    # 判断是否是叶子节点
    def isLeave(self):
        boo = self.left == None and self.right == None
        return boo

#create nodes创建叶子节点
def createNodes(tup):
    return [Node(freq[0],freq[1]) for freq in tup]

#create Huffman-Tree创建Huffman树
def createHuffmanTree(nodes):
    queue = nodes[:]
    while len(queue) > 1:
        queue.sort(key=lambda item:item.freq)
        node_left = queue.pop(0)
        node_right = queue.pop(0)
        node_father = Node(node_left.freq + node_right.freq,'')
        node_father.left = node_left
        node_father.right = node_right
        node_left.father = node_father
        node_right.father = node_father
        queue.append(node_father)
    queue[0].father = None
    return queue[0]

#Huffman解码，根据霍夫曼编码获取对应的字符。
def huffmanEncoding(nodes,root):
    codes = [''] * len(nodes)
    for i in range(len(nodes)):
        node_tmp = nodes[i]
        while node_tmp != root:
            if node_tmp.isLeft():
                codes[i] = '0' + codes[i]
            else:
                codes[i] = '1' + codes[i]
            node_tmp = node_tmp.father
    return codes

# root-root节点
# coder-霍夫曼编码，如1000100110
def huffmanDecoding(root,coder):
    for i in coder:
        if i == '1':
            root = root.right
        elif i == '0':
            root = root.left
        else:
            raise '霍夫曼编码不存在该字符' + i
        if root.isLeave():
            return root.char


if __name__ == '__main__':
    #chars = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
    #freqs = [10,4,2,5,3,4,2,6,4,4,3,7,9,6]
    chars_freqs = [('C', 2), ('G', 2), ('E', 3), ('K', 3), ('B', 4),
                   ('F', 4), ('I', 4), ('J', 4), ('D', 5), ('H', 6),
                   ('N', 6), ('L', 7), ('M', 9), ('A', 10)]
    nodes = createNodes([(item[1],item[0]) for item in chars_freqs])
    root = createHuffmanTree(nodes)
    codes = huffmanEncoding(nodes,root)

    for item in zip(chars_freqs,codes):
        print('Character:%s freq:%-2d   encoding: %s' % (item[0][0],item[0][1],item[1]))

    # char = huffmanDecoding(root, '000')
    char = huffmanDecoding(root, '11111')
    print('====',char)