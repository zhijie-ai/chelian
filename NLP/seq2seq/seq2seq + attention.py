#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/4 15:11                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf

batch_size = 5
encoder_input = tf.placeholder(shape=[batch_size, None], dtype=tf.int32 )
decoder_target = tf.placeholder(shape=[batch_size, None], dtype=tf.int32)

# Embedding
input_vocab_size = 10
target_vocab_size = 10
input_embedding_size = 20
target_embedding_size = 20

encoder_input_ = tf.one_hot(encoder_input,depth=input_vocab_size, dtype=tf.int32)
decoder_target_ = tf.one_hot(decoder_target,depth=target_vocab_size, dtype=tf.int32)

input_embedding = tf.Variable(tf.random_uniform(shape=[input_vocab_size, input_embedding_size],minval=-1.0,maxval= 1.0), dtype=tf.float32)
target_embedding = tf.Variable(tf.random_uniform(shape=[target_vocab_size, target_embedding_size], minval=-1.0, maxval=1.0), dtype=tf.float32)

input_embedd  = tf.nn.embedding_lookup(input_embedding, encoder_input)
target_embedd = tf.nn.embedding_lookup(target_embedding, decoder_target)

#Encoder
rnn_hidden_size = 20
cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_hidden_size)
init_state = cell.zero_state(batch_size, dtype=tf.float32)
encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, input_embedd, initial_state=init_state)

# Decoder
# helper
inference = False
seq_len = tf.constant([3,4,5,2,3], tf.int32)
if not inference:
    helper = tf.contrib.seq2seq.TrainingHelper(target_embedd, sequence_length=seq_len)
if inference:
    helper = tf.contrib.seq2seq.InferenceHelper(target_embedd, sequence_length=seq_len)

# rnn_cell
d_cell = tf.nn.rnn_cell.GRUCell(rnn_hidden_size)

#attention cell
attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_hidden_size, encoder_output)
decoder_cell = tf.contrib.seq2seq.AttentionWrapper(d_cell, attention_mechanism, attention_layer_size=rnn_hidden_size)
de_state = decoder_cell.zero_state(batch_size,dtype=tf.float32)
# OutputProjectionWrapper 相当于将decoder的输出加上一个dense层
out_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, target_vocab_size)

with tf.variable_scope('decoder'):
    decoder = tf.contrib.seq2seq.BasicDecoder(
    out_cell,
    helper,
    de_state,
    tf.layers.Dense(target_embedding_size))

# dynamic decoder
final_outputs, final_state, final_sequence_lengths = \
    tf.contrib.seq2seq.dynamic_decode(decoder,swap_memory=True)