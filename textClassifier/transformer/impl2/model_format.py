#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/3/2 8:52                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import tensorflow as tf
from utils import save_model

def format_saved_model(model_dir):
    sess = tf.compat.v1.Session()
    ckpt = tf.train.latest_checkpoint('log/1')
    saver = tf.compat.v1.train.import_meta_graph(ckpt+'.meta')
    saver.restore(sess,ckpt)

    graph = tf.get_default_graph()

    ds = graph.get_operation_by_name('IteratorGetNext').outputs
    y_hat = graph.get_tensor_by_name('y_hat:0')
    encoder_input = ds[0]
    decoder_input = ds[3]

    save_model(model_dir,sess,encoder_input,decoder_input,y_hat)

