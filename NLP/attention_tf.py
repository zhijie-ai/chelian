#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/7/8 17:37                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
#! -*- coding: utf-8 -*-

import tensorflow as tf
'''
inputs是一个形如(batch_size, seq_len, word_size)的张量；
函数返回一个形如(batch_size, seq_len, position_size)的位置张量。
'''
def Position_Embedding(inputs, position_size):
    batch_size,seq_len = tf.shape(inputs)[0],tf.shape(inputs)[1]
    position_j = 1. / tf.pow(10000.,
                             2 * tf.range(position_size / 2,dtype=tf.float32 ) / position_size)
    print('position_j.shape',position_j.shape)#(256,)
    position_j = tf.expand_dims(position_j, 0)
    print('position_j1.shape',position_j.shape)#(1, 256)
    position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
    print('position_i.shape',position_i.shape)#(50,)
    position_i = tf.expand_dims(position_i, 1)
    print('position_i2.shape',position_i.shape)#(50,1)
    position_ij = tf.matmul(position_i, position_j)#(50,256)
    position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)#(50,512)
    position_embedding = tf.expand_dims(position_ij, 0) \
                         + tf.zeros((batch_size, seq_len, position_size))
    print('tf.expand_dims(position_ij, 0).shape',tf.expand_dims(position_ij, 0).shape)#(1,50,512)
    print('tf.zeros((batch_size, seq_len, position_size)).shape',
          tf.zeros((batch_size, seq_len, position_size)).shape)
    return position_embedding

'''
inputs是一个二阶以上的张量，代表输入序列，比如形如(batch_size, seq_len, input_size)的张量；
seq_len是一个形如(batch_size,)的张量，代表每个序列的实际长度，多出部分都被忽略；
mode分为mul和add，mul是指把多出部分全部置零，一般用于全连接层之前；
add是指把多出部分全部减去一个大的常数，一般用于softmax之前。
'''
def Mask(inputs, seq_len, mode='mul'):
    if seq_len == None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_len), tf.float32)
        for _ in range(len(inputs.shape)-2):
            mask = tf.expand_dims(mask, 2)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12

'''
普通的全连接
inputs是一个二阶或二阶以上的张量，即形如(batch_size,...,input_size)。
只对最后一个维度做矩阵乘法，即输出一个形如(batch_size,...,ouput_size)的张量。
'''
def Dense(inputs, ouput_size, bias=True, seq_len=None):
    print('inputs.shape',inputs.shape)
    input_size = int(inputs.shape[-1])
    W = tf.Variable(tf.random_uniform([input_size, ouput_size], -0.05, 0.05))
    print('W.shape',W.shape)
    if bias:
        b = tf.Variable(tf.random_uniform([ouput_size], -0.05, 0.05))
    else:
        b = 0.0

    reshape_inputs = tf.reshape(inputs, (-1, input_size))
    print('reshape_inputs.shape',reshape_inputs.shape)
    outputs = tf.matmul(reshape_inputs, W) + b
    print('outputs.shape',outputs.shape)
    print('AAAAAAAAAA',tf.concat([tf.shape(inputs)[:-1], [ouput_size]], 0).shape)
    print('tf.shape(inputs)[:-1]',tf.shape(inputs)[:-1])
    outputs = tf.reshape(outputs,tf.concat([tf.shape(inputs)[:-1], [ouput_size]], 0))
    print('outputs1.shape',outputs.shape)
    if seq_len != None:
        outputs = Mask(outputs, seq_len, 'mul')
    return outputs

'''
Multi-Head Attention的实现
'''
def Attention(Q, K, V, nb_head, size_per_head, Q_len=None, V_len=None):
    #对Q、K、V分别作线性映射
    print('Q.shape',Q.shape)#(128, 50, 512)
    Q = Dense(Q, nb_head * size_per_head, False)
    print('Q1.shape',Q.shape)#(128, 50, 1024)
    Q = tf.reshape(Q, (-1, tf.shape(Q)[1], nb_head, size_per_head))
    print('Q2.shape', Q.shape)#(128, 50, 8, 128)
    Q = tf.transpose(Q, [0, 2, 1, 3])
    print('Q3.shape', Q.shape)#(128, 8,50, 128)
    K = Dense(K, nb_head * size_per_head, False)
    K = tf.reshape(K, (-1, tf.shape(K)[1], nb_head, size_per_head))
    K = tf.transpose(K, [0, 2, 1, 3])
    print('K.shape',K.shape)#(128, 8,50, 128)
    V = Dense(V, nb_head * size_per_head, False)
    V = tf.reshape(V, (-1, tf.shape(V)[1], nb_head, size_per_head))
    V = tf.transpose(V, [0, 2, 1, 3])
    print('V.shape',V.shape)#(128, 8,50, 128)
    #计算内积，然后mask，然后softmax
    A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(size_per_head))
    print('A.shape',A.shape)#(128, 8,50, 50)
    A = tf.transpose(A, [0, 3, 2, 1])
    print('A.shape', A.shape)#(128,50, 50, 8)
    A = Mask(A, V_len, mode='add')
    print('A.shape', A.shape)
    A = tf.transpose(A, [0, 3, 2, 1])
    print('A.shape', A.shape)
    A = tf.nn.softmax(A)
    print('A.shape', A.shape)
    #输出并mask
    O = tf.matmul(A, V)
    O = tf.transpose(O, [0, 2, 1, 3])
    O = tf.reshape(O, (-1, tf.shape(O)[1], nb_head * size_per_head))
    O = Mask(O, Q_len, 'mul')
    return O


if __name__ == '__main__':
    import numpy as np

    Q = np.random.randn(128,50,512)
    K = np.random.randn(128,50,512)
    V = np.random.randn(128,50,512)
    print(Q.dtype)

    Q = tf.convert_to_tensor(Q,dtype=tf.float32)
    K = tf.convert_to_tensor(K,dtype=tf.float32)
    V = tf.convert_to_tensor(V,dtype=tf.float32)
    print(Q)
    # O = Attention(Q,K,V,8,128)

    enc = Position_Embedding(Q,512)
    print(enc.shape)
