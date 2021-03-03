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
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.
Utility functions
'''

import tensorflow as tf
# from tensorflow.python import pywrap_tensorflow
# import numpy as np
import json
import os, re
import logging
from tensorflow.python.framework import graph_util

logging.basicConfig(level=logging.INFO)

def calc_num_batches(total_num, batch_size):
    '''Calculates the number of batches.
    total_num: total sample number
    batch_size
    Returns
    number of batches, allowing for remainders.'''
    return total_num // batch_size + int(total_num % batch_size != 0)

def convert_idx_to_token_tensor(inputs, idx2token):
    '''Converts int32 tensor to string tensor.
    inputs: 1d int32 tensor. indices.
    idx2token: dictionary
    Returns
    1d string tensor.
    '''
    def my_func(inputs):
        return " ".join(idx2token[elem] for elem in inputs)

    return tf.numpy_function(my_func, [inputs], tf.string)

# # def pad(x, maxlen):
# #     '''Pads x, list of sequences, and make it as a numpy array.
# #     x: list of sequences. e.g., [[2, 3, 4], [5, 6, 7, 8, 9], ...]
# #     maxlen: scalar
# #
# #     Returns
# #     numpy int32 array of (len(x), maxlen)
# #     '''
# #     padded = []
# #     for seq in x:
# #         seq += [0] * (maxlen - len(seq))
# #         padded.append(seq)
# #
# #     arry = np.array(padded, np.int32)
# #     assert arry.shape == (len(x), maxlen), "Failed to make an array"
#
#     return arry

def postprocess(hypotheses, idx2token):
    '''Processes translation outputs.
    hypotheses: list of encoded predictions
    idx2token: dictionary
    Returns
    processed hypotheses
    '''
    _hypotheses = []
    for h in hypotheses:
        sent = "".join(idx2token[idx] for idx in h)
        sent = sent.split("</s>")[0].strip()
        sent = sent.replace("▁", " ") # remove bpe symbols
        _hypotheses.append(sent.strip())
    return _hypotheses

def save_hparams(hparams, path):
    '''Saves hparams to path
    hparams: argsparse object.
    path: output directory.
    Writes
    hparams as literal dictionary to path.
    '''
    if not os.path.exists(path): os.makedirs(path)
    hp = json.dumps(vars(hparams))
    with open(os.path.join(path, "hparams"), 'w') as fout:
        fout.write(hp)

def load_hparams(parser, path):
    '''Loads hparams and overrides parser
    parser: argsparse parser
    path: directory or file where hparams are saved
    '''
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    d = open(os.path.join(path, "hparams"), 'r').read()
    flag2val = json.loads(d)
    for f, v in flag2val.items():
        parser.f = v

def save_variable_specs(fpath):
    '''Saves information about variables such as
    their name, shape, and total parameter number
    fpath: string. output file path
    Writes
    a text file named fpath.
    '''
    def _get_size(shp):
        '''Gets size of tensor shape
        shp: TensorShape
        Returns
        size
        '''
        size = 1
        for d in range(len(shp)):
            size *=shp[d]
        return size

    params, num_params = [], 0
    for v in tf.compat.v1.global_variables():
        params.append("{}==={}".format(v.name, v.shape))
        num_params += _get_size(v.shape)
    print("num_params: ", num_params)
    with open(fpath, 'w') as fout:
        fout.write("num_params: {}\n".format(num_params))
        fout.write("\n".join(params))
    logging.info("Variables info has been saved.")

def get_hypotheses(num_batches, num_samples, sess, tensor, dict):
    '''Gets hypotheses.
    num_batches: scalar.
    num_samples: scalar.
    sess: tensorflow sess object
    tensor: target tensor to fetch
    dict: idx2token dictionary
    Returns
    hypotheses: list of sents
    '''
    hypotheses = []
    for _ in range(num_batches):
        h = sess.run(tensor)
        hypotheses.extend(h.tolist())
    hypotheses = postprocess(hypotheses, dict)#将id变成token

    return hypotheses[:num_samples]

def calc_bleu(ref, translation):
    '''Calculates bleu score and appends the report to translation
    ref: reference file path
    translation: model output file path
    Returns
    translation that the bleu score is appended to'''
    get_bleu_score = "perl multi-bleu.perl {} < {} > {}".format(ref, translation, "temp")
    os.system(get_bleu_score)
    bleu_score_report = open("temp", "r").read()
    with open(translation, "a") as fout:
        fout.write("\n{}".format(bleu_score_report))
    try:
        score = re.findall("BLEU = ([^,]+)", bleu_score_report)[0]
        new_translation = translation + "B{}".format(score)
        os.system("mv {} {}".format(translation, new_translation))
        os.remove(translation)

    except: pass
    os.remove("temp")


# def get_inference_variables(ckpt, filter):
#     reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
#     var_to_shape_map = reader.get_variable_to_shape_map()
#     vars = [v for v in sorted(var_to_shape_map) if filter not in v]
#     return vars

# 将模型保存为savedModel格式，供tf serving调用
def save_model(saved_model_dir,sess,encoder_input,decoder_input,y_pred):
    # deocer_input = tf.ones((tf.shape(encoder_input)[0], 1), tf.int32) * 3 # 3代表<s>
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(saved_model_dir)
    inputs = {'encoder_input':tf.compat.v1.saved_model.utils.build_tensor_info(encoder_input),
              'decoder_input':tf.compat.v1.saved_model.utils.build_tensor_info(decoder_input)}
    outputs = {'y_pred':tf.compat.v1.saved_model.utils.build_tensor_info(y_pred)}
    signature = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(inputs,outputs,'test_sig_name')
    # builder.add_meta_graph_and_variables(sess,['test_saved_model'],{'test_signature':signature})
    builder.add_meta_graph_and_variables(sess,[tf.saved_model.SERVING],{'test_signature':signature})
    builder.save()

def load_saved_model(sess,dir):
    meta_graph_def = tf.compat.v1.saved_model.load(sess,[tf.saved_model.SERVING],dir)
    signature = meta_graph_def.signature_def

    # 从signature中找出具体输入输出的tensor name
    encoder_inp_name = signature['test_signature'].inputs['encoder_input'].name
    decoder_inp_name = signature['test_signature'].inputs['decoder_input'].name
    y_pred = signature['test_signature'].outputs['y_pred'].name

    encoder_input = sess.graph.get_tensor_by_name(encoder_inp_name)
    decoder_input = sess.graph.get_tensor_by_name(decoder_inp_name)
    y_pred = sess.graph.get_tensor_by_name(y_pred)
    # 在运行的时候，encoder和decoder的输入一定要和get_batche方法里的处理一样，先加结束符再定长处理
    return encoder_input,decoder_input,y_pred



# 将模型保存为pb格式
def save_model_to_pb(dir,sess):
    constant_graph = graph_util.convert_variables_to_constants(sess,sess.graph_def,['pb_model'])
    with tf.gfile.FastGFile(dir+'model.pb','wb') as f:
        f.write(constant_graph.SerializeToString())


def load_model_pb():
    model_path = 'XXX.pb'
    with tf.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='XXX')
    graph = tf.get_default_graph()
    input_images  = graph.get_tensor_by_name('XXX/image_tensor:0')
    output_num_boxes = graph.get_tensor_by_name('XXX/num_boxes:0')
    output_scores = graph.get_tensor_by_name('XXX/scores:0')

