#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/2/1 22:13                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# 使用spark中LR训练模型,这次模型应该也是要定时定时训练更新的，由于在课程中写在jupyter notebook中的，是为了展示
import os
import sys
# 如果当前代码文件运行测试需要加入修改路径，避免出现后导包问题
BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))# reco_sys
sys.path.insert(0, os.path.join(BASE_DIR))

PYSPARK_PYTHON = "/miniconda2/envs/reco_sys/bin/python"
# 当存在多个版本时，不指定很可能会导致出错
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_PYTHON


from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel
from offline import SparkSessionBase

class CtrLogisticRegression(SparkSessionBase):
    SPARK_APP_NAME='ctrLogisticRegression'
    ENABLE_HIVE_SUPPORT=True

    def __init__(self):
        self.spark = self._create_spark_hbase()


ctr = CtrLogisticRegression()

# 1.进行行为日志数据读取
ctr.spark.sql('use profile')
news_article_basic = ctr.spark.sql('select user_id,article_id,clicked from  user_article_basic')
news_article_basic.show()
# +-------------------+-------------------+-------+
# |            user_id|         article_id|clicked|
# +-------------------+-------------------+-------+
# |1105045287866466304|              14225|  false|
# |1106476833370537984|              14208|  false|
# |1109980466942836736|              19233|  false|
# |1109980466942836736|              44737|  false|
# |1109993249109442560|              17283|  false|
# |1111189494544990208|              19322|  false|
# |1111524501104885760|              44161|  false|

# 2. 用户画像数据读取与日志数据合并
user_profile_hbase = ctr.spark.sql('select user_id,information.gender,information.birthday,article_partial from user_profile_hbase')
user_profile_hbase.show()
# +--------------------+------+--------+--------------------+
# |             user_id|gender|birthday|     article_partial|
# +--------------------+------+--------+--------------------+
# |              user:1|  null|     0.0|Map(18:vars -> 0....|
# |             user:10|  null|     0.0|Map(18:tp2 -> 0.2...|
# |             user:11|  null|     0.0|               Map()|
# |user:110249052282...|  null|     0.0|               Map()|
# |user:110319567345...|  null|    null|Map(18:Animal -> ...|
# |user:110504528786...|  null|    null|Map(18:text -> 0....|
# |user:110509388310...|  null|    null|Map(18:text -> 0....|
# |user:110510518565...|  null|    null|Map(18:SHOldboySt...|
# 对于用户id做一个处理，取出前面的user字符串
def get_user_id(row):
    return int(row.user_id.split(':')[1]),row.gender,row.birthday,row.article_partial

# 错误,由于gender中存在大量连续的空值，spark无法自动确定类型,因此必须手动指定类型
# user_profile_hbase = user_profile_hbase.rdd.map(deal_with_user_id).toDF(['user_id','gender','birthday','article_partial'])
user_profile = user_profile_hbase.rdd.map(get_user_id)

_schema=StringType([
    StructField('user_id',LongType()),
    StructField('gender',BooleanType()),
    StructField('birthday',DoubleType()),
    StructField('article_partial',MapType(StringType(),DoubleType()))
])
user_profile_hbase = ctr.spark.createDataFrame(user_profile,schema=_schema).drop(['gender','birthday'])
user_profile_hbase
# DataFrame[user_id: bigint, article_partial: map<string,double>]

train = news_article_basic.join(user_profile_hbase,'user_id','left')
train.show()
# +-------------------+----------+-------+--------------------+
# |            user_id|article_id|clicked|     article_partial|
# +-------------------+----------+-------+--------------------+
# |1106473203766657024|     16005|  false|Map(18:text -> 0....|
# |1106473203766657024|     17665|  false|Map(18:text -> 0....|
# |1106473203766657024|     44664|  false|Map(18:text -> 0....|
# |1106473203766657024|     44386|  false|Map(18:text -> 0....|

# 文章频道与向量读取合并，删除无用的特征，合并文章画像的权重特征
ctr.spark.sql('use article')
article_vector= ctr.spark.sql('select * from article_vector')
train_user_article = train.join(article_vector,'article_id','left')
train_user_article.show()
# +----------+-------------------+-------+--------------------+----------+--------------------+
# |article_id|            user_id|clicked|     article_partial|channel_id|       articlevector|
# +----------+-------------------+-------+--------------------+----------+--------------------+
# |     13401|1114864237131333632|  false|Map(18:vars -> 0....|        18|[0.06157120217893...|
# |     13401|                 10|  false|Map(18:tp2 -> 0.2...|        18|[0.06157120217893...|
# |     13401|1106396183141548032|  false|Map(18:tp2 -> 0.2...|        18|[0.06157120217893...|
# |     13401|1109994594201763840|  false|Map(18:tp2 -> 0.2...|        18|[0.06157120217893...|

# 读取文章画像
ctr.spark.sql('use profile')
article_profile = ctr.spark.sql('select article_id,keywords from article_profile')
# 处理文章权重
def get_article_weights(row):
    try:
        weights = sorted(row.keywords.values())[0:10]
    except Exception as e:
        # 给定异常默认值
        weights = [0.0] *10

    return row.article_id,weights

article_profile = article_profile.rdd.map(get_article_weights).toDF(['article_id','article_weights'])

# article_profile
train_user_article = train_user_article.join(article_profile,'article_id','left')
train_user_article.show()
# +----------+-------------------+-------+--------------------+----------+--------------------+--------------------+
# |article_id|            user_id|clicked|     article_partial|channel_id|       articlevector|     article_weights|
# +----------+-------------------+-------+--------------------+----------+--------------------+--------------------+
# |     13401|1114864237131333632|  false|Map(18:vars -> 0....|        18|[0.06157120217893...|[0.08196639249252...|
# |     13401|                 10|  false|Map(18:tp2 -> 0.2...|        18|[0.06157120217893...|[0.08196639249252...|
train_user_article
# DataFrame[article_id: bigint, user_id: bigint, clicked: boolean, article_partial: map<string,double>, channel_id: int, articlevector: array<double>, article_weights: array<double>]
# 4. 进行用户的权重特征筛选处理，类型处理
train_user_article  = train_user_article.dropna()
columns = ['article_id','user_id','channel_id','articlevector',
           'user_weights','article_weights','clicked']
# array->vector
def get_user_weights(row):
    # 取出所有对应的partial频道的关键词权重(用户)
    from pyspark.ml.linalg import Vectors

    try:
        weights = sorted([row.article_partial[key] for key in row.article_partial.keys() if key.split(':')[0]==str(row.channel_id)])[:10]
    except Exception as e:
        weights = [0.0] * 10

    return row.article_id,row.user_id,row.channel_id,Vectors.dense(row.articlevector),Vectors.dense(weights),Vectors.dense(row.article_weights),int(row.clicked)

train_vector = train_user_article.rdd.map(get_user_weights).toDF(columns)

train_vector
# DataFrame[article_id: bigint, user_id: bigint, channel_id: bigint, articlevector: vector, user_weights: vector, article_weights: vector, clicked: bigint]
# 使用收集特征到features [channel_id,articlevector,user_weights,article_weights]
train_res = VectorAssembler().setInputCols(columns[2:6]).setOutputCol('features').transform(train_vector)
# features 121值, 13, 18,       1,2,3,4,5,6....25
# 25 + 100 + 10 + 10 = 145个特征
train_res.show()
# +----------+-------------------+----------+--------------------+-------420471378...|[0.08196639249252...|      0|[18.0,0.061571202...|-------------+--------------------+-------+--------------------+
# |article_id|            user_id|channel_id|       articlevector|        user_weights|     article_weights|clicked|            features|
# +----------+-------------------+----------+--------------------+--------------------+--------------------+-------+--------------------+
# |     13401|1114864237131333632|        18|[0.06157120217893...|[0.32473

# 处理要写入的训练样本格式
train = train_res.select(['article_id','user_id','clicked','feature'])
arr = train.collect()

# 处理DataFrame Pandas
import pandas as pd
df = pd.DataFrame(arr)

# 生成tfrecords样本格式数据
import tensorflow as tf

def write_to_tfrecords(click_batch,feature_batch):
    '''
    将用户与文章的点击日志构造的样本写入TFRecords文件
    :param click_batch:
    :param feature_batch:
    :return:
    '''
    # 1、构造tfrecords的存储实例
    writer = tf.python_io.TFRecordWriter('./train_ctr_20190523.tfrecords')

    # 2、循环将所有样本一个个封装成example，写入这个文件
    for i in range(len(click_batch)):
        # 取出第i个样本的目标值与特征值，格式转换
        click = click_batch[i]
        feature = feature_batch[i].tostring()

        # 构造example，int64,float64,bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            'label':tf.train.Feature(int64_list = tf.train.Int64List(value=[click])),
            'features':tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature]))
        }))
        # 序列化example，写入文件
        writer.write(example.SerializeToString())
    writer.close()

# 开启会话打印内容
with tf.Session() as sess:
    # 创建线程协调器
    coord = tf.train.Coordinator()

    # 开启子线程去读取数据
    # 返回子线程实例
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    # 存入数据
    write_to_tfrecords(df.iloc[:,2],df.iloc[:,3])

    # 关闭子线程，回收
    coord.request_stop()
    coord.join(threads)