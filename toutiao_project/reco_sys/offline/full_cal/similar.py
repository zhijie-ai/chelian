#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/1/13 10:46                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# 文章相似度的计算,用历史数据来计算，因此只需计算运行一次即可，后面的增量更新时需要定时运行的
import os
import sys
sys.path.insert(0,'/usr/local/service/spark/python/lib/pyspark.zip')
sys.path.insert(0,'/usr/local/service/spark/python/lib/py4j-0.10.7-src.zip')
sys.path.insert(0,'/usr/local/service/spark/jars/spark-core_2.11-2.3.2.jar')

BASE_DIR = os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(BASE_DIR))
from offline import SparkSessionBase
from setting.default import CHANNEL_INFO
from pyspark.ml.feature import Word2Vec


class TrainWord2VecModel(SparkSessionBase):
    SPARK_APP_NAME = "Word2Vec"
    SPARK_URL = "yarn"

    ENABLE_HIVE_SUPPORT = True

    def __init__(self):
        self.spark = self._create_spark_session()


w2v = TrainWord2VecModel()

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


w2v.spark.sql('use article')
article = w2v.spark.sql('select * from article_data where channel_id =18')
# **Spark Word2Vec训练保存模型**

words_df = article.rdd.mapPartitions(segmentation).toDF(['article_id', 'channel_id', 'words'])
new_word2Vec = Word2Vec(vectorSize=100, inputCol="words", outputCol="model", minCount=3)
new_model = new_word2Vec.fit(words_df)
# new_model.save("hdfs:///recommendation/models/{}_{}.word2vec".format(channel_id,channel_name))
vectors = new_model.getVectors()

# from pyspark.ml.feature import Word2VecModel
channel_id = 18
channel = "python"
# wv_model = Word2VecModel.load(
#                 "hdfs:///recommendation/models/{}_{}.word2vec".format(channel_id,channel))
# vectors = wv_model.getVectors()




# 获取文章画像数据，得到文章画像的关键词
profile = w2v.spark.sql('select * from article_profile where channel_id={}'.format(channel_id))
profile.registerTempTable("incremental")
articleKeywordsWeights = w2v.spark.sql('select article_id,channel_id,key_word,weight from incremental LATERAL VIEW explode(keyword) AS key_word,weight')

_article_profile = articleKeywordsWeights.join(vectors,vectors.word ==articleKeywordsWeights.key_word,'inner')

# 计算得到文章每个词的向量
articleKeywordsVectors = _article_profile.rdd.map(lambda row:(row.article_id,row.channel_id,row.key_word,
                                     row.weight*row.vector)).toDF(["article_id", "channel_id", "keyword", "weightVector"])

# 计算得到文章的平均词向量即文章的向量
def avg(row):
    x=0
    for v in row.vectors:
        x+=v
        # 将平均向量作为article的向量
    return row.article_id,row.channel_id,x/len(row.vectors)

articleKeywordsVectors.registerTempTable('tempTable')
articleVector = w2v.spark.sql(
    'select article_id,min(channel_id) channel_id,collect_set(weightVector) vectors from tempTable group by article_id').rdd.map(avg).toDF(["article_id", "channel_id", "articleVector"]
)

# 对计算出的”articleVector“列进行处理，该列为Vector类型，不能直接存入HIVE，HIVE不支持该数据类型
def toArray(row):
    return row.article_id,row.channel_id,[float(i) for i in row.articleVector.toArray()]
articleVector = articleVector.rdd.map(toArray).toDF(['article_id','channel_id','articleVector'])
# 最终计算出这个18号Python频道的所有文章向量，保存到固定的表当中
# articleVector.write.insertInto("article_vector")

# 文章相似度计算
# 计算每个频道两两文章的相似度，并保存
# 1. 可以对每个频道内N个文章聚成M类别，那么类别数越多每个类别的文章数量越少。如下pyspark代码
# from pyspark.ml.clustering import BisectingKMeans
# bkmeans = BisectingKMeans(k=100, minDivisibleClusterSize=50, featuresCol="articleVector", predictionCol='group')
# bkmeans_model = bkmeans.fit(articleVector)
# bkmeans_model.save(
#     "hdfs:////recommendation/models/modelsbak/articleBisKmeans/channel_%d_%s.bkmeans" % (channel_id, channel))

# 但是对于每个频道聚成多少类别这个M是超参数，并且聚类算法的时间复杂度并不小，
# 当然可以使用一些优化的聚类算法二分、层次聚类。

# 局部敏感哈希LSH(Locality Sensitive Hashing)
from pyspark.ml.linalg import Vectors
# 选取部分数据做测试
article_vector = w2v.spark.sql('select article_id,articlevector from article_vector where channel_id=18 limit 5')
train = article_vector
def _array_to_vector(row):
    return row.article_id,Vectors.dense(row.articlevector)
train = train.rdd.map(_array_to_vector).toDF(['article_id','articleVector'])

# BRP进行FIT
from pyspark.ml.feature import BucketedRandomProjectionLSH

brp = BucketedRandomProjectionLSH(inputCol = 'articleVector',outputCol = 'hashes',
                                  numHashTables=4.0,bucketLength=10.0)
model = brp.fit(train)
# 计算相似的文章以及相似度
similarity = model.approxSimilarityJoin(train,train,2.0,distCol='EuclideanDistance')
similarity.sort(['EuclideanDistance']).show()


# 保存到hbase
def save_habase(partition):
    import happybase
    pool = happybase.ConnectionPool(size=3,host='10.0.80.13')
    with pool.connection() as conn:
        table = conn.table('article_similar')
        for row in partition:
            if row.datasetA.article_id ==row.datasetB.article_id:



                pass
            else:
                table.put(str(row.datasetA.article_id).encode(),{
                    'similar:{}'.format((row.datasetB.article_id).encode()): b'%0.4f' % (row.EuclideanDistance)
                })

# similarity.foreachPartition(save_habase)