#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/9/28 9:30
 =================知行合一=============
'''
import requests
from bs4 import BeautifulSoup



id = '932'
url = 'https://www.julyedu.com/question/big/kp_id/26/ques_id/'
header = {'User-Agent': 'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)'}
resp = requests.get(url+id).text
soup = BeautifulSoup(resp, 'lxml')

def fun(tag):
    return not tag.has_attr('src') and tag.name == 'script'

print(len(soup.find_all(fun)))
script_code = soup.find_all(fun)[2].get_text()
print(script_code)


#获取script标签中的数据，因为加载的数据都在script中


def parse_question_and_answer(script_code):
    import json

    json_data = script_code.split('}]};')[0]+'}]}'
    json_data = json_data[15:]
    json_data.replace('\r','')
    data_dict  = json.loads(json_data)
    question_answer = data_dict['quesInfo']
    questions = data_dict['list']
    ques = question_answer['ques']
    answer = question_answer['analysis']
    return ques,answer

def getAllQuestions(script_code):
    import json

    q=[]
    json_data = script_code.split('}]};')[0] + '}]}'
    explaintion = json_data[15:]
    data_dict = json.loads(explaintion)
    questions = data_dict['list']
    for dict in questions:
        q.append(dict['ques_id'])
    return q

def get_question(id,ind):
    url = 'https://www.julyedu.com/question/big/kp_id/23/ques_id/'+str(id)
    # header = {'User-Agent': 'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)'}
    resp = requests.get(url).text
    soup = BeautifulSoup(resp, 'lxml')
    script_code = soup.find_all(fun)[3].get_text()
    name,answer = parse_question_and_answer(script_code)
    with open('question2.txt','a+',encoding='utf-8') as file:
        file.write('{}. {}'.format(ind,name)+'\n')
        file.write('答案链接：{}\n'.format(url))
        # file.write(answer+'\n')


if __name__ == '__main__':
    print(parse_question_and_answer(script_code))

    ques_ids = getAllQuestions(script_code)
    print(ques_ids)
    # for ind,id in enumerate(ques_ids):
    #     get_question(id,(ind+1))
