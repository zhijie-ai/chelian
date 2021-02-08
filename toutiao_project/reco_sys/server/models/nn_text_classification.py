#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/2/12 12:09                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
from tensorflow import keras

vocab_size=5000
max_sentence=200

def get_train_test():
    import numpy as np
    '''
    获取电影评论文本数据
    :return:
    '''
    imdb = keras.datasets.imdb

    (x_train_source,y_train),(x_test_source,y_test) = imdb.load_data(num_words=vocab_size)
    res = []
    for i in range(len(x_train_source)):
        res.extend(x_train_source[i])

    res=set(res)
    res2 = set(range(1,5001))
    print(res2-res)

    # 每个样本评论序列长度固定
    x_train = keras.preprocessing.sequence.pad_sequences(x_train_source,
                                                         maxlen=max_sentence,
                                                         padding='post',value=0)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test_source,
                                                        maxlen=max_sentence,
                                                        padding='post', value=0)
    return (x_train,y_train),(x_test,y_test)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = get_train_test()

    def parser(x,y):
        features={'feature':x}
        print(features)
        return features,y

    def train_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
        dataset = dataset.shuffle(buffer_size=25000)
        dataset = dataset.batch(64)
        dataset = dataset.map(parser)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


    def eval_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        dataset = dataset.batch(64)
        dataset = dataset.map(parser)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    # 2、 模型输入特征列指定
    column = tf.feature_column.categorical_column_with_identity('feature',vocab_size)
    # 词向量长度大小
    embedding_column = tf.feature_column.embedding_column(column,dimension=50)

    # 3 、模型训练与保持
    classifier = tf.estimator.DNNClassifier(
        hidden_units=[100],
        feature_columns=[embedding_column],
        model_dir='./ckpt/text_embedding'
    )

    classifier.train(input_fn=train_input_fn)
    result = classifier.evaluate(input_fn=eval_input_fn)
    print(result)