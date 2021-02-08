#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/9/28 9:30
 =================知行合一=============
'''
import requests
from bs4 import BeautifulSoup




url = 'https://www.julyedu.com/question/big/kp_id/23/ques_id/919'
header = {'User-Agent':'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)'}
resp = requests.get(url=url,headers = header).text
soup = BeautifulSoup(resp,'html.parser')

# print(resp)
#
import json
#
# data = json.loads(resp)
# questions = data['data']
# print(questions[0])
# print(questions[0]['ques'])

def fun(tag):
    return not tag.has_attr('src') and tag.name == 'script'

data = soup.find_all(fun)[3].get_text()
data = data.split('}]};')[0]+'}]}'
data = data[15:]
print(data)

data_dict  = json.loads(data)
question_answer = data_dict['quesInfo']
questions = data_dict['list']
print(questions[0])
ques = question_answer['ques']
answer = question_answer['analysis']