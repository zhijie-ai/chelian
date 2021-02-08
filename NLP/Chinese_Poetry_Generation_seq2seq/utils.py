#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/4 16:52                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
#! /usr/bin/env python3
# -*- coding:utf-8 -*-


def is_cn_char(ch):
    """ Test if a char is a Chinese character. """
    return ch >= u'\u4e00' and ch <= u'\u9fa5'

def is_cn_sentence(sentence):
    """ Test if a sentence is made of Chinese characters. """
    for ch in sentence:
        if not is_cn_char(ch):
            return False
    return True

def split_sentences(text):
    """ Split a piece of text into a list of sentences. """
    sentences = []
    i = 0
    for j in range(len(text) + 1):
        if j == len(text) or \
                text[j] in [u'，', u'。', u'！', u'？', u'、', u'\n']:
            if i < j:
                sentence = u''.join(filter(is_cn_char, text[i:j]))
                sentences.append(sentence)
            i = j + 1
    return sentences

NUM_OF_SENTENCES = 4
CHAR_VEC_DIM = 512