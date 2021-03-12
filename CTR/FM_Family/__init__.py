#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/3/12 16:46                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
'''
FM原理中，计算的是Vi和Vj的点积。注意外层的2个求和。经过化简，可以变成和的平方减去平方的和，此时
    得到一个向量，外层再reduce_sum即变为一个标量。
NFM:中间经过Bi-interaction pooling得到一个向量。注意，是此条训练集中value为1的特征对应的embedding进行
    element-wis product操作。经过化简可以变成和FM公式类似，只是少了一层求和
AFM:pair-wise interaction layer虽然和Bi-interaction 很像，但是前者输出多个向量，后者输出一个向量。差了一个求和的操作


如果最外层加一个求和，则FM原始公式(vi与vj的点积),bi-interaction,pair-wise interaction得到的结果是一样的。
    FM可以化简成reduce_sum(和的平方减去平方的和),bi-interaction也可以化简成和的平方减去平方的和，pair-wise interaction得到的是多个向量，
    如果不考虑Attention机制，在Pair-wise Interaction Layer之后直接得到最终输出，当p全为1时，就是FM，
    不加attention，就是将多个向量求和，得到一个向量，再求和就变成了FM。
'''