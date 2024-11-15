#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/21 19:51                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
from __future__ import print_function
from textClassifier.transformer.impl.hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex

def load_de_vocab():
    # vocab已经包含了4个特殊字符
    vocab = [line.split()[0] for line in codecs.open('preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

# 把每行数据增加一个结束符并定长处理
def create_data(source_sents, target_sents):
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()

    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [de2idx.get(word, 1) for word in (source_sent + u" </S>").split()]  # 1: OOV, </S>: End of Text
        y = [en2idx.get(word, 1) for word in (target_sent + u" </S>").split()]
        if max(len(x), len(y)) <= hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)

    # Pad
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        # np.lib.pad：表示各个方向填充的个数(0,4),表示在第一个轴前填充0个，后填充4个，第二个轴前0个，后4个
        # ((2,4),(1,3)),表示第一个周前2个，后4个，第二个轴前1个后3个
        X[i] = np.lib.pad(x, [0, hp.maxlen - len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen - len(y)], 'constant', constant_values=(0, 0))

    # X:里面是词对应的id，Sources：里面是词
    return X, Y, Sources, Targets



def load_train_data():
    # 把训练集变成一个数组
    de_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in
                codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    en_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in
                codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]

    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Y


def load_test_data():
    def _refine(line):
        line = regex.sub("<[^>]+>", "", line)
        line = regex.sub("[^\s\p{Latin}']", "", line)
        return line.strip()

    de_sents = [_refine(line) for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n") if
                line and line[:4] == "<seg"]
    en_sents = [_refine(line) for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n") if
                line and line[:4] == "<seg"]

    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Sources, Targets  # (1064, 150)


def get_batch_data():
    # Load data
    X, Y = load_train_data()  # 加了eos并定长处理

    # calc total batch count
    num_batch = len(X) // hp.batch_size

    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)

    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])

    # create batch queues,tf.train.batch也是放入一个 queue runner
    x, y = tf.train.shuffle_batch(input_queues,
                                  num_threads=8,
                                  batch_size=hp.batch_size,
                                  capacity=hp.batch_size * 64,
                                  min_after_dequeue=hp.batch_size * 32,
                                  allow_smaller_final_batch=False)

    return x, y, num_batch  # (N, T), (N, T), ()

if __name__ == '__main__':
    print(load_de_vocab()[0])