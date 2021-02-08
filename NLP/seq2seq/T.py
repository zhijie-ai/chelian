#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/15 15:41                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np

dec_output = np.random.uniform(size=(10,20,128))
enc_output = np.random.uniform(size=(10,20,128))

shuf = np.expand_dims(dec_output,-1)
print('shuf.shape',shuf.shape)#(10, 20, 128, 1)
enc_output_shuf = np.tile(np.expand_dims(enc_output,1),[1,20,1,1]) #(batch,len,len,enc_units)
print('np.expand_dims(enc_output,1).shape',np.expand_dims(enc_output,1).shape)#(10, 1, 20, 128)
print('enc_output_shuf.shape',enc_output_shuf.shape)#(10, 20, 20, 128)
res = np.matmul(enc_output_shuf,shuf)
print('res.shape',res.shape)#(10, 20, 20, 1)
print('np.transpose(enc_output_shuf,[0,1,3,2]).shape',np.transpose(enc_output_shuf,[0,1,3,2]).shape)
#(10, 20, 128, 20)

# soft = np.nn.softmax(np.matmul(enc_output_shuf,shuf))#(batch,len,len,1)
# enc_out_shuf = np.transpose(enc_output_shuf,[0,1,3,2])#(batch,len,enc_units,len)
# attention = np.matmul(enc_out_shuf,soft)#(batch,len,enc_units,1)
# fin_attention = np.squeeze(attention,-1) #(batch,len,enc_units)