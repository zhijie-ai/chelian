# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2021/3/4 14:56                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------

# https://blog.csdn.net/maqian5/article/details/108404175
import requests
import time

t1 = time.time()
encoder_inp = [[1, 2, 3, 0, 0], [2, 3, 4, 0, 0]]
decoder_inp = [[3], [3]]

dit = {'signature_name': 'test_signature',
       'inputs': {'encoder_input': encoder_inp,
                  'decoder_input': decoder_inp}}

header = {'User-Agent':
              'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24'}

resp = requests.post('http://localhost:8501/v1/models/transformer_model:predict', json=dit)
print(resp.text)
t2 = time.time()
print('time cost:{} s'.format(t2-t1))
