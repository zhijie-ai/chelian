#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/11/28 17:14                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    return serialized_example

def read_batch(file_name, max_epoch, batch_size, thread_num, min_after_dequeue):
    with tf.name_scope("input"):
        filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(file_name),
                                            num_epochs=max_epoch)
        serialized_example = read_and_decode(filename_queue)
        capacity = thread_num * batch_size + min_after_dequeue
        batch_serialized_example = tf.train.shuffle_batch(
                                    [serialized_example],
                                    batch_size=batch_size,
                                    num_threads=thread_num,
                                    capacity=capacity,
                                    min_after_dequeue=min_after_dequeue)
        features = tf.parse_example(
                    batch_serialized_example,
                    features={
                            "label": tf.FixedLenFeature([], tf.float32),
                            "ids": tf.FixedLenFeature([39], tf.int64),
                            "values": tf.FixedLenFeature([39], tf.float32),
                            })
        return features["label"], features["ids"], features["values"]