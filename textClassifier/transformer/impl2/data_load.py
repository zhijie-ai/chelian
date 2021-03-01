#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2021/2/21 21:32                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
Note.
if safe, entities on the source side have the prefix 1, and the target side 2, for convenience.
For example, fpath1, fpath2 means source file path and target file path, respectively.
'''
import tensorflow as tf
from utils import calc_num_batches

def load_vocab(vocab_fpath):
    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <s>, 3: </s>
    Returns
    two dictionaries.
    '''
    vocab = [line.split()[0] for line in open(vocab_fpath, 'r',encoding='utf8').read().splitlines()]
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    return token2idx, idx2token

def load_data(fpath1, fpath2, maxlen1, maxlen2):
    '''Loads source and target data and filters out too lengthy samples.
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.
    Returns
    sents1: list of source sents
    sents2: list of target sents
    '''
    sents1, sents2 = [], []
    with open(fpath1, 'r',encoding='utf8') as f1, open(fpath2, 'r',encoding='utf8') as f2:
        for sent1, sent2 in zip(f1, f2):
            if len(sent1.split()) + 1 > maxlen1: continue # 1: </s>
            if len(sent2.split()) + 1 > maxlen2: continue  # 1: </s>
            sents1.append(sent1.strip())
            sents2.append(sent2.strip())
    return sents1, sents2


def encode(inp, type, dict):
    '''Converts string to number. Used for `generator_fn`.
    inp: 1d byte array.
    type: "x" (source side) or "y" (target side)
    dict: token2idx dictionary
    Returns
    list of numbers
    '''
    inp_str = inp.decode("utf-8")#如果是字符串则似乎是二进制数据
    if type=="x": tokens = inp_str.split() + ["</s>"]
    else: tokens = ["<s>"] + inp_str.split() + ["</s>"]

    x = [dict.get(t, dict["<unk>"]) for t in tokens]
    return x

def generator_fn(sents1, sents2, vocab_fpath):
    '''Generates training / evaluation data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.
    yields
    xs: tuple of
        x: list of source token ids in a sent
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) sentence
    labels: tuple of
        decoder_input: decoder_input: list of encoded decoder inputs
        y: list of target token ids in a sent
        y_seqlen: int. sequence length of y
        sent2: str. target sentence
    '''
    token2idx, _ = load_vocab(vocab_fpath)
    for sent1, sent2 in zip(sents1, sents2):
        x = encode(sent1, "x", token2idx)#如果是source则添加一个结束符
        y = encode(sent2, "y", token2idx)# 如果是target这添加开始和结束符
        decoder_input, y = y[:-1], y[1:]

        x_seqlen, y_seqlen = len(x), len(y)
        yield (x, x_seqlen, sent1), (decoder_input, y, y_seqlen, sent2)

def input_fn(sents1, sents2, vocab_fpath, batch_size, shuffle=False):
    '''Batchify data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean
    Returns
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    '''
    #tf.TensorShape([]) 表示长度为单个数字
    #tf.TensorShape([None]) 表示长度未知的向量
    shapes = (([None], (), ()),
              ([None], [None], (), ()))
    types = ((tf.int32, tf.int32, tf.string),
             (tf.int32, tf.int32, tf.int32, tf.string))
    paddings = ((0, 0, ''),
                (0, 0, 0, ''))

    dataset = tf.data.Dataset.from_generator(
        generator_fn,# 该生成器的作用就是给X添加结束符，给decoder部分添加开始和结束符。
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2, vocab_fpath))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset

def get_batch(fpath1, fpath2, maxlen1, maxlen2, vocab_fpath, batch_size, shuffle=False):
    '''Gets training / evaluation mini-batches
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean
    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    '''
    # source,target
    sents1, sents2 = load_data(fpath1, fpath2, maxlen1, maxlen2)
    batches = input_fn(sents1, sents2, vocab_fpath, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(sents1), batch_size)
    return batches, num_batches, len(sents1)