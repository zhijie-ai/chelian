#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/5/23 16:38                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
from GAN.bigGAN.utils import *

data = load_mnist()
inputs = tf.data.Dataset.from_tensor_slices(data)
inputs_iterator = inputs.make_one_shot_iterator()

inputs2 = inputs_iterator.get_next()
print(inputs2)

data = tf.data.Dataset.from_tensor_slices(data)
data = data.batch(10)
print(data)
iterator = tf.data.Iterator.from_structure(data.output_types,data.output_shapes)
init_op = iterator.make_initializer(data)
with tf.Session()  as sess:
    sess.run(init_op)
try:
    images, labels = iterator.get_next()
except tf.errors.OutOfRangeError:
    sess.run(init_op)