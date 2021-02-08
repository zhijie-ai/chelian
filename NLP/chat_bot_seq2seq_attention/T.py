#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/4 21:42                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
from NLP.chat_bot_seq2seq_attention.data_loader import loadDataset,getBatches

data_path = 'data/dataset-cornell-length10-filter1-vocabSize40000.pkl'
word2id, id2word, trainingSamples = loadDataset(data_path)
print(trainingSamples[0:2])
batches = getBatches(trainingSamples, 4)
print(batches[0].encoder_inputs)
print(batches[0].encoder_inputs_length)
print(batches[0].decoder_targets)
print(batches[0].decoder_targets_length)
print(word2id['<eos>'])