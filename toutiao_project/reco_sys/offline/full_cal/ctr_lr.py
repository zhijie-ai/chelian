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
user_article_basic = ctr.spark.sql('select user_id,article_id,clicked')
user_article_basic.show()
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
def deal_with_user_id(row):
    return int(row.user_id.split(':')[1]),row.gender,row.birthday,row.article_partial

# 错误,由于gender中存在大量连续的空值，spark无法自动确定类型,因此必须手动指定类型
# user_profile_hbase = user_profile_hbase.rdd.map(deal_with_user_id).toDF(['user_id','gender','birthday','article_partial'])
user_profile = user_profile_hbase.rdd.map(deal_with_user_id)

_schema=StringType([
    StructField('user_id',LongType()),
    StructField('gender',BooleanType()),
    StructField('birthday',DoubleType()),
    StructField('article_partial',MapType(StringType(),DoubleType()))
])
user_profile_hbase = ctr.spark.createDataFrame(user_profile,schema=_schema).drop(['gender','birthday'])
user_profile_hbase
# DataFrame[user_id: bigint, article_partial: map<string,double>]

train = user_article_basic.join(user_profile_hbase,'user_id','left')
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
train = train.join(article_vector,'article_id','left')
train.show()
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
train = train.join(article_profile,'article_id','left')
train.show()
# +----------+-------------------+-------+--------------------+----------+--------------------+--------------------+
# |article_id|            user_id|clicked|     article_partial|channel_id|       articlevector|     article_weights|
# +----------+-------------------+-------+--------------------+----------+--------------------+--------------------+
# |     13401|1114864237131333632|  false|Map(18:vars -> 0....|        18|[0.06157120217893...|[0.08196639249252...|
# |     13401|                 10|  false|Map(18:tp2 -> 0.2...|        18|[0.06157120217893...|[0.08196639249252...|
train
# DataFrame[article_id: bigint, user_id: bigint, clicked: boolean, article_partial: map<string,double>, channel_id: int, articlevector: array<double>, article_weights: array<double>]
# 4. 进行用户的权重特征筛选处理，类型处理
train  = train.dropna()
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

train_1 = train.rdd.map(get_user_weights).toDF(columns)

train_1
# DataFrame[article_id: bigint, user_id: bigint, channel_id: bigint, articlevector: vector, user_weights: vector, article_weights: vector, clicked: bigint]
# 使用收集特征到features [channel_id,articlevector,user_weights,article_weights]
train_vector_two = VectorAssembler().setInputCols(columns[2:6]).setOutputCol('features').transform(train_1)
# features 121值, 13, 18,       1,2,3,4,5,6....25
# 25 + 100 + 10 + 10 = 145个特征
train_vector_two.show()
# +----------+-------------------+----------+--------------------+-------420471378...|[0.08196639249252...|      0|[18.0,0.061571202...|-------------+--------------------+-------+--------------------+
# |article_id|            user_id|channel_id|       articlevector|        user_weights|     article_weights|clicked|            features|
# +----------+-------------------+----------+--------------------+--------------------+--------------------+-------+--------------------+
# |     13401|1114864237131333632|        18|[0.06157120217893...|[0.32473

lr = LogisticRegression()
model = lr.setLabelCol('clicked').setFeaturesCol('features').fit(train_vector_two)
model.save("hdfs://hadoop-master:9000/headlines/models/test_ctr.obj")

online_model = LogisticRegressionModel.load('hdfs://hadoop-master:9000/headlines/models/test_ctr.obj')
res_transform = online_model.transform(train_vector_two)

def vector_to_double(row):
    return float(row.clicked),float(row.probability[1])

score_label = res_transform.select(['clicked','probability']).rdd.map(vector_to_double)

# 模型评估-Accuracy与AUC
import matplotlib.pyplot as plt
plt.figure()
plt.plot([0,1],[0,1],'r--')
plt.plot(model.summary.roc.select('FPR').collect(),
         model.summary.roc.select('TPR').collect())
plt.xlabel('FPR')
plt.ylabel('TPR')

# AUC值计算
from pyspark.mllib.evaluation import BinaryClassificationMetrics
metrics = BinaryClassificationMetrics(score_label)
metrics.areaUnderROC
# 0.7364334522585716

# 其他方法
from sklearn.metrics import roc_auc_score,accuracy_score
import numpy as np
arr = np.array(score_label.collect())
accuracy_score(arr[:, 0], arr[:, 1].round())
# 0.9051438053097345
roc_auc_score(arr[:, 0], arr[:, 1])
# 0.719274521004087



##################################################
# clicked:目标值
# probability:[不点击的概率，点击的概率]
ctr.spark.sql('use profile')
user_profile_hbase = ctr.spark.sql(
    'select user_id,information.birthday,information.gender,article_partial from user_profile_hbase'
)

# 对特征工程的处理，特征中心的更新,即特征服务中心
# 抛弃获取值少的特征
user_profile_hbase = user_profile_hbase.drop(['birthday','gender'])

def get_user_id(row):
    return int(row.user_id.split(':')[1]),row.article_partial

user_profile_hbase_temp = user_profile_hbase.rdd.map(get_user_id)

from pyspark.sql.types import *

_schema = StructType([
    StructField("user_id", LongType()),
    StructField("weights", MapType(StringType(), DoubleType()))
])
user_profile_hbase_schema = ctr.spark.createDataFrame(user_profile_hbase_temp,schema=_schema)

def feature_preprocess(row):
    from pyspark.ml.linalg import Vectors

    channel_weights=[]
    for i in range(1,26):
        try:
            _res = sorted([row.weights[key] for key in row.weights.keys() if key.split(':')[0]==str(i)])[:10]
            channel_weights.append(_res)
        except Exception as e:
            channel_weights.append([0.0]*10)
    return row.user_id,channel_weights

res = user_profile_hbase_schema.rdd.map(feature_preprocess).collect()

#
import happybase
# 批量插入hbase数据库中
pool = happybase.ConnectionPool(size=10,host='hadoop-master',port=9090)
with pool.connection() as conn:
    ctr_feature_user = conn.table('ctr_feature_user')
    with ctr_feature_user.batch(transaction=True) as b:
        for i in range(len(res)):
            for j in range(25):
                b.put('{}'.format(res[i][0]).encode(),
                      {'channel:{}'.format(j+1).encode():str(res[i][1][j]).encode()})


# 文章特征中心更新
ctr.spark.sql('use article')
article_profile  = ctr.spark.sql('select * from article_profile')
# 进行文章相关特征处理和提取
def article_profile_to_feature(row):
    try:
        weights = sorted(row.keywords.values())[:10]
    except Exception as e:
        weights = [0.0]*10
    return row.article_id,row.channel_id,weights

article_profile = article_profile.rdd.map(article_profile_to_feature).toDF(['article_id','channel_id','weights'])
article_profile.show()

# 文章向量特征的提取
article_vector = ctr.spark.sql('select * from article_vector')
article_feature = article_profile.join(article_vector,'article_id','inner')
def feature_to_vector(row):
    from pyspark.ml.linalg import Vectors
    return row.article_id,row.channel_id,Vectors.dense(row.weights),Vectors.dense(row.articlevector)
article_feature = article_feature.rdd.map(feature_to_vector).toDF(['article_id','channel_id','weights','articlevector'])

# 指定所有文章特征进行合并
cols2 = ['article_id','channel_id','weights','articlevector']
article_feature_two = VectorAssembler().setInputCols(cols2[1:4]).setOutputCol('features').transform(article_feature)
# +----------+----------+--------------------+--------------------+--------------------+
# |article_id|channel_id|             weights|       articlevector|            features|
# +----------+----------+--------------------+--------------------+--------------------+
# |        26|        17|[0.19827163395829...|[0.02069368539384...|[17.0,0.198271633...|
# |        29|        17|[0.26031398249056...|[-0.1446092289546...|[17.0,0.260313982...|

# 保存到特征数据库中
def save_article_feature_to_hbase(partition):
    import happybase
    pool = happybase.ConnectionPool(size=10,host='hadoop-master',port=9090)
    with pool.connection() as conn:
        table = conn.table('ctr_feature_article')
        for row in partition:
            table.put('{}'.format(row.article_id).encode(),
                      {'article:{}'.format(row.article_id).encode():str(row.features).encode()})

article_feature_two.foreachPartition(save_article_feature_to_hbase)
