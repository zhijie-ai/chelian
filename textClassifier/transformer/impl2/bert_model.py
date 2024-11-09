# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/9/19 13:15                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # 加载base模型的对应的切词器
print(tokenizer) # 打印出对应的信息，如base模型的字典大小，截断长度等等
token = tokenizer.tokenize("Don't Make sUch a fuss,get StUFf!") # 切词
print(token) # 切词结果
indexes = tokenizer.convert_tokens_to_ids(token) # 将词转换为对应字典的id
print(indexes) # 输出id
tokens = tokenizer.convert_ids_to_tokens(indexes)# 将id转换为对应字典的词
print(tokens) # 输出词

# 使用这种方法对句子编码会自动添加[CLS] 和[SEP]
input_ids = torch.tensor(tokenizer.encode("Don't Make sUch a fuss,get StUFf!")).squeeze(0)
print(input_ids)
cls_id = tokenizer._convert_token_to_id('[CLS]')
sep_id = tokenizer._convert_token_to_id('[SEP]')
print(cls_id, sep_id)

import torch
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
input_ids = torch.tensor(tokenizer.encode("Don't Make sUch a fuss,get StUFf!")).unsqueeze(0)  # Batch size 1
input_ids2 = torch.tensor(tokenizer.encode("how old you")).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
outputs2 = model(input_ids2)
print(outputs[0].shape) # 词的向量表示
print(outputs[1].shape) # 句子的向量表示

print(outputs2[0].shape)
print(outputs2[1].shape)

