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
FM的求解可以用化简公式，也可以用IFM中的思路，循环遍历i和j最终得到结果。

FM原理中，计算的是Vi和Vj的点积。注意外层的2个求和。经过化简，可以变成和的平方减去平方的和，此时
    得到一个向量，外层再reduce_sum即变为一个标量。
NFM:中间经过Bi-interaction pooling得到一个向量。注意，是此条训练集中value为1的特征对应的embedding进行
    element-wis product操作。经过化简可以变成和FM公式类似，只是少了一层求和
AFM:pair-wise interaction layer虽然和Bi-interaction 很像，但是前者输出多个向量，后者输出一个向量。差了一个求和的操作

如果最外层加一个求和，则FM原始公式(vi与vj的点积),bi-interaction,pair-wise interaction得到的结果是一样的。
    FM可以化简成reduce_sum(和的平方减去平方的和),bi-interaction也可以化简成和的平方减去平方的和，pair-wise interaction得到的是多个向量，
    如果不考虑Attention机制，在Pair-wise Interaction Layer之后直接得到最终输出，当p全为1时，就是FM，
    不加attention，就是将多个向量求和，得到一个向量，再求和就变成了FM。

NFM的Bilinear layer就是在做二阶特征组合。最终可以化简为和的平方减去平方的和，最外层没有求和，所以是一个向量。
AFM如果没有权重aij,那么他就是一个FM算法，为了得到aij，需要使用bi-interaction的思路得到多个向量，再根据这多个向量
    求aij，如果直接借鉴deepFM的lookup的思路，虽然可以得到多个embedding向量，但是并没有交叉特征的意义在里面，仅仅是原始的
    向量，且少于AFM的思路得到的向量的个数。

IFM的特征处理方式有点特别,输入是libsvm格式的数据
dict中的key是libsvm中的16:1(index为16，值为1)。合起来为key
AFM的输入是csv格式的数据,将train合test的数据concat，区分连续值类型合category类型，依次将每一个特征分配一个index
然后给出对应df的index及value，类别特征为1，数值型特征为对应的value
'''
import tensorflow as tf

tf.keras.applications.NASNetLarge