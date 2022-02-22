#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/9/9 15:19                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.python.feature_column.feature_column import _LazyBuilder

def test_numeric():
    price={'price':[[1.0],[2.0],[3.0],[4.0]]}
    builder = _LazyBuilder(price)

    def transform_fn(x):
        return x+2

    price_column = feature_column.numeric_column('price',normalizer_fn=transform_fn)
    # price_transformed_tensor = price_column._get_dense_tensor(builder)
    #
    # with tf.Session() as session:
    #     print(session.run([price_transformed_tensor]))

    # 使用input_layer
    price_transformed_tensor = tf.compat.v1.feature_column.input_layer(price, [price_column])

    with tf.compat.v1.Session() as session:
        print('use input_layer' + '_' * 40)
        print(session.run([price_transformed_tensor]))


if __name__ == '__main__':
    test_numeric()