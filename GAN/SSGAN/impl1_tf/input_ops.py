#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/5/7 15:18                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np
import tensorflow as tf

from GAN.SSGAN.impl1_tf.util import log


def check_data_id(dataset,data_id):
    if not data_id:
        return

    wrong = []
    for id in data_id:
        if id in dataset.data:
            pass
        else:
            wrong.append(id)

    if len(wrong) >0:
        raise RuntimeError('There are %d invalid ids,including %s'%(len(wrong),wrong[:5]))

def create_input_ops(dataset,
                     batch_size,
                     num_threads=16,# for creating batches
                     is_training=False,
                     data_id=None,
                     scope='inputs',
                     shuffle=True):
    '''
    Return a batched tensor for the inputs from the dataset
    :param dataset:
    :param batch_size:
    :param num_threads:
    :param is_training:
    :param data_id:
    :param scope:
    :param shuffle:
    :return:
    '''
    input_ops={}

    if data_id is None:
        data_id=dataset.ids
        log.info("input_ops [%s]: Using %d IDs from dataset", scope, len(data_id))
    else:
        log.info("input_ops [%s]: Using specified %d IDs", scope, len(data_id))

    # single operations
    with tf.device('/cpu:0'),tf.name_scope(scope):
        input_ops['id']=tf.train.string_input_producer(
            tf.convert_to_tensor(data_id),capacity=128
        ).dequeue(name='input_ids_dequeue')

        m,label = dataset.get_data(data_id[0])

        def load_fn(id):
            # image [n,n],label:[m]
            image,label = dataset.get_data(id)
            return (id,image.astype(np.float32),label.astype(np.float32))

        input_ops['id'],input_ops['image'],input_ops['label']=tf.py_func(
            load_fn,inp=[input_ops['id']],
            Tout=[tf.string,tf.float32,tf.float32],
            name='func_hp'
        )

        input_ops['id'].set_shape([])
        input_ops['image'].set_shape(list(m.shape))
        input_ops['label'].set_shape(list(label.shape))

    # batchify
    capacity = 2*batch_size*num_threads
    min_capacity = min(int(capacity*0.75),1024)

    if shuffle:
        batch_ops = tf.train.shuffle_batch(
            input_ops,
            batch_size=batch_size,
            num_threads=num_threads,
            min_after_dequeue=min_capacity,
            capacity=capacity
        )
    else:
        batch_ops = tf.train.batch(
            input_ops,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity
        )

    return input_ops,batch_ops
