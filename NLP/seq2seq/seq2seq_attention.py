#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/4 11:18                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf

batch = 128
enc_vocab = 100
enc_dim = 50
dec_vocab = 200
dec_dim = 100
enc_units = 128
dec_units = 128
max_len = 50



raw_enc_input = tf.placeholder(tf.int32,(batch,None))
raw_dec_input = tf.placeholder(tf.int32,(batch,None))
raw_output = tf.placeholder(tf.int32,(batch,None))
enc_sen = tf.placeholder(tf.int32,(None))
dec_sen = tf.placeholder(tf.int32,(None))

###EMBED
##encoder
enc_embed = tf.Variable(tf.random_uniform((enc_vocab,enc_dim),-1,1))
enc_input = tf.nn.embedding_lookup(enc_embed,raw_enc_input)
ch_enc_input = tf.layers.dense(enc_input,enc_dim,use_bias=False)

##decoder
dec_embed = tf.Variable(tf.random_uniform((dec_vocab,dec_dim),-1,1))
dec_input = tf.nn.embedding_lookup(dec_embed,raw_dec_input)
ch_dec_input = tf.layers.dense(dec_input,dec_dim,use_bias=False)

#real output
query_lis = [1.0] * dec_vocab
query_embed = tf.Variable(tf.matrix_diag(query_lis),trainable=False)
real_output = tf.nn.embedding_lookup(query_embed,raw_output)

###RNN_CELL
##encoder
with tf.name_scope('encoder'):
    enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicRNNCell(enc_units) for _ in range(3)])
    #enc_cell = tf.nn.rnn_cell.BasicRNNCell(enc_units)
    enc_h0 = enc_cell.zero_state(batch,tf.float32)
    enc_output,enc_out_state = tf.nn.dynamic_rnn(enc_cell,ch_enc_input,initial_state=enc_h0,sequence_length=enc_sen,scope='encoder')
#enc_output -> (batch,len,enc_units)

##decoder
with tf.name_scope('decoder'):
    dec_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicRNNCell(dec_units) for _ in range(3)])
    #dec_cell = tf.nn.rnn_cell.BasicRNNCell(dec_units)
    dec_h0 = dec_cell.zero_state(batch,tf.float32)
    dec_output,dec_out_state = tf.nn.dynamic_rnn(dec_cell,ch_dec_input,initial_state=enc_out_state,sequence_length=dec_sen,scope='decoder')
#dec_out_state -> (batch,dec_units)


###ATTENTION
shuf = tf.expand_dims(dec_output,-1) #(batch,len,enc_units,1)
enc_output_shuf = tf.tile(tf.expand_dims(enc_output,1),[1,max_len,1,1]) #(batch,len,len,enc_units)
soft = tf.nn.softmax(tf.matmul(enc_output_shuf,shuf))#(batch,len,len,1)
enc_out_shuf = tf.transpose(enc_output_shuf,[0,1,3,2])#(batch,len,enc_units,len)
attention = tf.matmul(enc_out_shuf,soft)#(batch,len,enc_units,1)
fin_attention = tf.squeeze(attention,-1) #(batch,len,enc_units)

###concat
fin_out = tf.tanh(tf.concat((fin_attention,dec_output),-1)) #(batch,len,2 * enc_units)

###ln
# mean, variance = tf.nn.moments(fin, [-1], keep_dims=True)
# normalized = (fin - mean) / ((variance + 0.000001) ** (.5))
# fin_out = tf.layers.dense(
#     normalized,(dec_units+enc_units),
#     kernel_initializer = tf.initializers.zeros(),
#     bias_initializer = tf.initializers.ones(),
#     activation=None
# )

#output
output = tf.layers.dense(fin_out,
                         dec_vocab,
                         activation = None,
                         bias_initializer = tf.initializers.zeros(),
                         kernel_initializer = tf.initializers.random_uniform(-0.1,0.1),
                         kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0003),
)



loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output,labels=real_output),-1)+ tf.add_n(tf.get_collection('regularization_losses'))


train = tf.train.AdamOptimizer(0.001).minimize(loss)