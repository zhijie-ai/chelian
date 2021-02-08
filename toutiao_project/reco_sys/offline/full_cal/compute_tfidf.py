#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/1/10 17:47                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import os
import sys
import gc

sys.path.insert(0,'/usr/local/service/spark/python/lib/pyspark.zip')
sys.path.insert(0,'/usr/local/service/spark/python/lib/py4j-0.10.7-src.zip')
sys.path.insert(0,'/usr/local/service/spark/jars/spark-core_2.11-2.3.2.jar')
# 如果当前代码文件运行测试需要加入修改路径，避免出现后导包问题
BASE_DIR = os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(BASE_DIR))
from offline import SparkSessionBase


class KeywordsToTfidf(SparkSessionBase):
    SPARK_APP_NAME = "keywordsByTFIDF"
    #SPARK_EXECUTOR_MEMORY = "7g"

    ENABLE_HIVE_SUPPORT = True

    def __init__(self):
        self.spark = self._create_spark_session()


def segmentation(partition):
    import os
    import re

    import jieba
    import jieba.analyse
    import jieba.posseg as pseg
    import codecs

    # abspath = '/recommendation/words/ITKeywords.txt'
    # jieba.load_userdict(abspath)

    # 停用词
    # stopwords_path = os.path.join(abspath, "stopwords.txt")
    # def get_stopwords_list():
    #     stopwords_list = [i.strip() for i in codecs.open(stopwords_path).readlines()]
    #     return stopwords_list

    def cut_sentence(sentence):
        '''对切割之后的词语进行过滤，去除停用词，保留名词，英文和自定义词库中的词，长度大于2的词'''
        seg_list = pseg.lcut(sentence)
        # seg_list = [i for i in seg_list if i.flag not in stopwords_list]#i.word,i.flag
        filtered_words_list=[]
        for seg in seg_list:
            if len(seg.word) <=1:
                continue
            elif seg.flag=='eng':
                if len(seg.word)<=2:
                    continue
                else:
                    filtered_words_list.append(seg.word)
            elif seg.flag.startswith("n"):
                filtered_words_list.append(seg.word)
            elif seg.flag in ["x", "eng"]:  # 是自定一个词语或者是英文单词
                filtered_words_list.append(seg.word)
        return filtered_words_list

    for row in partition:
        sentence = re.sub("<.*?>", "", row.sentence)
        words  = cut_sentence(sentence)
        yield row.article_id,row.channel_id,words


ktt = KeywordsToTfidf()

ktt.spark.sql('use article')
article_dataframe = ktt.spark.sql("select * from article_data")
words_df = article_dataframe.rdd.mapPartitions(segmentation).toDF(['article_id','channel_id','words'])

# 词语与词频统计
from pyspark.ml.feature import CountVectorizer
cv = CountVectorizer(inputCol='words',outputCol='countFeatures',vocabSize=200*10000,minDF=1.0)
# 训练词频统计模型
cv_model = cv.fit(words_df)
cv_model.write().overwrite().save('hdfs:///recommendation/models/CV.model')
# cv_model.vocabulary
# 得出词频向量结果
cv_result = cv_model.transform(words_df)

# 训练idf模型，保存
from pyspark.ml.feature import CountVectorizerModel
# cv_model = CountVectorizerModel.load("hdfs:///recommendation/models/CV.model")
# 得出词频向量结果
# cv_result = cv_model.transform(words_df)# 不需要通过CountVectorizerModel来重新加载模型的
# 训练IDF模型
from pyspark.ml.feature import IDF
idf = IDF(inputCol='countFeatures',outputCol='idfFeatures')
idf_model = idf.fit(cv_result)
# idfModel.idf.toArray()#得到idf的值，transform得到tfidf的值
idf_model.write().overwrite().save("hdfs:///recommendation/models/IDF.model")

# from pyspark.ml.feature import IDFModel
# idfModel = IDFModel.load("hdfs:///recommendation/models/IDF.model")

# cv_model有word，idf_model中index，idf值
# [(word,idf),(word,idf)]
keywords_list_with_idf = list(zip(cv_model.vocabulary, idf_model.idf.toArray()))
def func(data):
    for index in range(len(data)):
        data[index] = list(data[index])
        data[index].append(index)
        data[index][1] = float(data[index][1])
func(keywords_list_with_idf)
sc = ktt.spark.sparkContext
rdd = sc.parallelize(keywords_list_with_idf)
df = rdd.toDF(["keywords", "idf", "index"])
df.write.insertInto('idf_keywords_values')


# cv_model = CountVectorizerModel.load("hdfs:///recommendation/models/CV.model")
# idf_model = IDFModel.load("hdfs:///recommendation/models/IDF.model")
cv_result = cv_model.transform(words_df)
tfidf_result = idf_model.transform(cv_result)

def func(partition):
    TOPK = 20
    for row in partition:
        # 找到索引与IDF值并进行排序
        _ = list(zip(row.idfFeatures.indices, row.idfFeatures.values))
        _ = sorted(_, key=lambda x: x[1], reverse=True)
        result = _[:TOPK]
        for word_index, tfidf in result:
            yield row.article_id, row.channel_id, int(word_index), round(float(tfidf), 4)

# index,tfidf
_keywordsByTFIDF = tfidf_result.rdd.mapPartitions(func).toDF(["article_id", "channel_id", "index", "tfidf"])

# 利用结果索引与”idf_keywords_values“合并知道词
keywordsIndex = ktt.spark.sql("select keyword, index idx from idf_keywords_values")
# 利用结果索引与”idf_keywords_values“合并知道词
keywordsByTFIDF = _keywordsByTFIDF.join(keywordsIndex, keywordsIndex.idx == _keywordsByTFIDF.index).select(["article_id", "channel_id", "keyword", "tfidf"])
keywordsByTFIDF.write.insertInto("tfidf_keywords_values")

del keywordsIndex
del keywordsByTFIDF
gc.collect()

# 2.4.3 TextRank提取关键词
# 计算textrank
def textrank(partition):
    import os
    import jieba
    import jieba.analyse
    import jieba.posseg as pseg
    import codecs

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
        tags = textrank_model.textrank(row.sentence,topK=20,
                                       withWeight=True,allowPOS=allowPOS,withFlag=False)
        for tag in tags:
            yield row.article_id, row.channel_id, tag[0], tag[1]


textrank_keywords_df = article_dataframe.rdd.mapPartitions(textrank).toDF(
["article_id", "channel_id", "keyword", "textrank"])
textrank_keywords_df.write.insertInto("textrank_keywords_values")


# 2.5.3 文章画像结果
# - 步骤：
#   - 1、加载IDF，保留关键词以及权重计算(TextRank * IDF)
#   - 2、合并关键词权重到字典结果
#   - 3、将tfidf和textrank共现的词作为主题词
#   - 4、将主题词表和关键词表进行合并，插入表

# 加载IDF，保留关键词以及权重计算(TextRank * IDF)
idf = ktt.spark.sql("select * from idf_keywords_values")
idf = idf.withColumnRenamed("keyword", "keyword1")
result = textrank_keywords_df.join(idf,textrank_keywords_df.keyword==idf.keyword1)
# 每个词的idf值*textrank值
keywords_res = result.withColumn("weights",
                                 result.textrank * result.idf).select(["article_id",
                                                                       "channel_id",
                                                                       "keyword", "weights"])
# 合并关键词权重到字典结果
keywords_res.registerTempTable("temptable")
merge_keywords = ktt.spark.sql("select article_id, min(channel_id) channel_id, collect_list(keyword) keywords, collect_list(weights) weights from temptable group by article_id")
# 合并关键词权重合并成字典
def _func(row):
    return row.article_id,row.channel_id,dict(zip(row.keywords,row.weights))
keywords_info = merge_keywords.rdd.map(_func).toDF(["article_id", "channel_id", "keywords"])

# 将tfidf和textrank共现的词作为主题词
topic_sql = """
                select t.article_id article_id2, collect_set(t.keyword) topics from tfidf_keywords_values t
                inner join 
                textrank_keywords_values r
                where t.keyword=r.keyword
                group by article_id2
                """
article_topics = ktt.spark.sql(topic_sql)

# 将主题词表和关键词表进行合并
article_profile = keywords_info.join(article_topics, keywords_info.article_id==article_topics.article_id2).select(["article_id", "channel_id", "keywords", "topics"])
article_profile.write.insertInto("article_profile")

# 原始文章表数据合并得到文章所有的词语句信息