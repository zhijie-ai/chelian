#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/2/11 13:07                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import functools

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

train_file = "G:\\workspace\\chelian\\toutiao_project\\reco_sys\\server\\models\data\\adult.data"
test_file = "G:\\workspace\\chelian\\toutiao_project\\reco_sys\\server\\models\data\\adult.test"

def input_func(file,epoches,batch_size):
    '''
    解析普查数据csv格式样本
    :param file:
    :param epoches:
    :param batch_size:
    :return:
    '''
    def deal_with_csv(value):
        data = tf.decode_csv(value,record_defaults=_CSV_COLUMN_DEFAULTS)

        # 构建列名称与这一行值的字典数据
        feature_dict = dict(zip(_CSV_COLUMNS,data))
        labels = feature_dict.pop('income_bracket')
        classes = tf.equal(labels,'>50K')
        return feature_dict,classes

    # 1、读取美国普查收入样本数据
    # tensorflow的迭代，一行样本数据
    # 名称要指定
    # 39,State-gov,77516,Bachelors,13,,Adm-clerical
    dataset = tf.data.TextLineDataset(file)# Dataset的子类实现
    dataset = dataset.map(deal_with_csv)
    # dataset， 包含了feature_dict, classes， 迭代器
    dataset = dataset.repeat(epoches)
    dataset = dataset.batch(batch_size)
    return dataset

def get_feature_column():
    '''
    指定输入estimator中特征列类型
    :return:
    '''
    # 数值型特征
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    numeric_columns = [age,education_num,capital_gain,capital_loss,hours_per_week]

    # 类别特征
    # categorical_column_with_vocabulary_list:将字符串转换成ID
    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship',
        ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative']
    )
    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    # categorical_column_with_hash_bucket->哈希列
    # 对不确定类别数量以及字符串时，哈希列进行分桶
    occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation',
                                                                       hash_bucket_size=1000)

    categorical_columns = [relationship,marital_status,workclass,occupation]

    return numeric_columns+categorical_columns

def get_feature_column_v2():
    '''
    特征交叉与分桶
    :return:
    '''
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    numeric_columns = [age, education_num, capital_gain, capital_loss, hours_per_week]

    # 类别型特征
    # categorical_column_with_vocabulary_list, 将字符串转换成ID
    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship',
        ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    # categorical_column_with_hash_bucket--->哈希列
    # 对不确定类别数量以及字符时，哈希列进行分桶
    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=1000)
    categorical_columns = [relationship, marital_status, workclass, occupation]

    # 分桶，交叉特征
    age_buckets = tf.feature_column.bucketized_column(
        age,boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    )
    crossed_columns = [
        tf.feature_column.crossed_column(
            ['education','occupation'],hash_bucket_size=1000
        ),
        tf.feature_column.crossed_column(
            [age_buckets,'education','occupation'],hash_bucket_size=1000
        )
    ]

    return numeric_columns+categorical_columns+crossed_columns



if __name__ == '__main__':
    dataset = input_func(train_file, 3, 32)
    # 构造模型
    feature_cl = get_feature_column()
    classification = tf.estimator.LinearClassifier(feature_columns=feature_cl)
    # train输入的input_func，不能调用传入
    # 1、input_func，构造的时候不加参数，但是这样不灵活， 里面参数不能固定的时候
    # 2、functools.partial
    train_func = functools.partial(input_func,train_file,epoches=3,batch_size=32)
    test_func = functools.partial(input_func,test_file,epoches=1,batch_size=32)
    classification.train(train_func)
    result = classification.evaluate(test_func)
    print(result)

    # 分桶交叉特征之后的效果
    feature_v2 = get_feature_column_v2()
    classifiry = tf.estimator.LinearClassifier(feature_columns=feature_v2)
    # train输入的input_func，不能调用传入
    # 1、input_func，构造的时候不加参数，但是这样不灵活， 里面参数不能固定的时候
    # 2、functools.partial
    train_func = functools.partial(input_func, train_file, epoches=3, batch_size=32)
    test_func = functools.partial(input_func, test_file, epoches=1, batch_size=32)
    classifiry.train(train_func)# 也可以通过lambda表达式的方式传参
    result=classifiry.evaluate(test_func)
    print(result)