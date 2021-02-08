#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/1/11 16:47                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

def textrank(partition):
    import jieba.analyse

    abspath = "/root/words"

    # # 结巴加载用户词典
    # userDict_path = os.path.join(abspath, "ITKeywords.txt")
    # jieba.load_userdict(userDict_path)
    #
    # # 停用词文本
    # stopwords_path = os.path.join(abspath, "stopwords.txt")
    #
    # def get_stopwords_list():
    #     """返回stopwords列表"""
    #     stopwords_list = [i.strip()
    #                       for i in codecs.open(stopwords_path).readlines()]
    #     return stopwords_list
    # # 所有的停用词列表
    # stopwords_list = get_stopwords_list()

    class TextRank(jieba.analyse.TextRank):
        def __init__(self,window=20,word_min_len=2):
            super(TextRank,self).__init__()
            self.span = window
            self.word_min_len = word_min_len
            # 要保留的词性，根据jieba github ，具体参见https://github.com/baidu/lac
            self.pos_fit = frozenset(('n', 'x', 'eng', 'f', 's', 't', 'nr', 'ns',
                                      'nt', "nw", "nz", "PER", "LOC", "ORG"))
        def pairfilter(self,wp):
            '''过滤条件，返回True或者False'''
            if wp.flag=='eng':
                if len(wp.word)<=2:
                    return False
            if wp.flag in self.pos_filt and len(wp.word.strip()) >= self.word_min_len: #and wp.word.lower() not in stopwords_list:
                return True

    # TextRank过滤窗口大小为5，单词最小为2
    textrank_model = TextRank(window=5,word_min_len=2)
    allowPOS = ('n', "x", 'eng', 'nr', 'ns', 'nt', "nw", "nz", "c")

    for row in partition:
        tags = textrank_model.textrank(row,topK=20,
                                       withWeight=True,
                                       allowPOS=allowPOS,withFlag=False)
        for tag in tags:
            yield tag[0], tag[1]

data = [line.split('~')[1] for  line in open('000000_0')]

print(list(textrank(data)))

