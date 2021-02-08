#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/4 16:56                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from NLP.Chinese_Poetry_Generation_seq2seq.generate import Generator
from NLP.Chinese_Poetry_Generation_seq2seq.plan import Planner


if __name__ == '__main__':
    planner = Planner()
    generator = Generator()
    while True:
        hints = input("Type in hints >> ")
        keywords = planner.plan(hints)
        print("Keywords: " + ' '.join(keywords))
        poem = generator.generate(keywords)
        print("Poem generated:")
        for sentence in poem:
            print(sentence)