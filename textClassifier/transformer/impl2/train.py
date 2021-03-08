#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2021/2/21 21:35                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
import tensorflow as tf

from model import Transformer
from tqdm import tqdm
from data_load import get_batch
from utils import save_hparams, save_variable_specs, get_hypotheses, calc_bleu,format_saved_model,save_model
import os
from hparams import Hparams
import math
import logging


logging.basicConfig(level=logging.INFO)

os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)

logging.info("# Prepare train/eval batches")
train_batches, num_train_batches, num_train_samples = get_batch(hp.train1, hp.train2,
                                             hp.maxlen1, hp.maxlen2,
                                             hp.vocab, hp.batch_size,
                                             shuffle=True)
eval_batches, num_eval_batches, num_eval_samples = get_batch(hp.eval1, hp.eval2,
                                             100000, 100000,
                                             hp.vocab, hp.batch_size,
                                             shuffle=False)

# create a iterator of the correct shape and type
# 或者用dataset.make_one_shoe_iterator来构造迭代器,可以直接用for i in dataset来遍历，也可以构造一个迭代器
iter = tf.compat.v1.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(train_batches),
                                       tf.compat.v1.data.get_output_shapes(train_batches))
xs, ys = iter.get_next()

train_init_op = iter.make_initializer(train_batches)#字符串似乎是二进制数据
eval_init_op = iter.make_initializer(eval_batches)

logging.info("# Load model")
m = Transformer(hp)
loss, train_op, global_step, train_summaries = m.train(xs, ys)
y_hat, eval_summaries = m.eval(xs, ys)
# y_hat = m.infer(xs, ys)

logging.info("# Session")
saver = tf.compat.v1.train.Saver(max_to_keep=hp.num_epochs)
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)#按需设置显存
with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options,
                                                          allow_soft_placement=True,
                                                          log_device_placement=True)) as sess:
    ckpt = tf.compat.v1.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.compat.v1.global_variables_initializer())
        save_variable_specs(os.path.join(hp.logdir, "specs"))
    else:
        saver.restore(sess, ckpt)

    summary_writer = tf.compat.v1.summary.FileWriter(hp.logdir, sess.graph)

    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches#122700
    _gs = sess.run(global_step)#如果重复训练，global_step为122701
    for i in tqdm(range(_gs, total_steps+1)):
        _, _gs, _summary = sess.run([train_op, global_step, train_summaries])
        epoch = math.ceil(_gs / num_train_batches)
        summary_writer.add_summary(_summary, _gs)

        if _gs and _gs % num_train_batches == 0:
            logging.info("epoch {} is done".format(epoch))
            _loss = sess.run(loss) # train loss

            logging.info("# test evaluation")
            _, _eval_summaries = sess.run([eval_init_op, eval_summaries])
            summary_writer.add_summary(_eval_summaries, _gs)

            logging.info("# get hypotheses")
            hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat, m.idx2token)#将id->token

            logging.info("# write results")
            model_output = "iwslt2016_E%02dL%.2f" % (epoch, _loss)
            if not os.path.exists(hp.evaldir): os.makedirs(hp.evaldir)
            translation = os.path.join(hp.evaldir, model_output)
            with open(translation, 'w') as fout:
                fout.write("\n".join(hypotheses))

            logging.info("# calc bleu score and append it to translation")
            calc_bleu(hp.eval3, translation)

            logging.info("# save models")
            ckpt_name = os.path.join(hp.logdir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)

            logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            logging.info("# fall back to train mode")
            sess.run(train_init_op)

    #保存为saved_model格式的模型，供tf serving使用，如果放在for循环里，
    # 每次都保存一个模型同saver.save方法，虽然精确到秒不会冲突，但saved_model模型应该是最后一个或者最优的模型
    saved_model_dir='./saved_model'
    # ckpt_dir = hp.logdir
    # format_saved_model(ckpt_dir,saved_model_dir)
    save_model(saved_model_dir,sess,xs[0],ys[0],y_hat)
    summary_writer.close()


logging.info("Done")