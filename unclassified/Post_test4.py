#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/9/20 11:59                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

import requests
import json

import json

import requests

app_id = "623896511427641344"
app_key = "277b42a2be4244a3aa4451520afc8ad6"
fee_type = 0

data_file = 'data.txt'
log_file = 'result.txt'

url = "http://api.unidt.com/api/fractal/openid/report?app_id=%s&app_key=%s&fee_type=0" % (app_id, app_key)

data = {
    # "image": "base64",
    "image_url": "http://mmbiz.qpic.cn/mmbiz_png/UtWdDgynLdbIjl7VGqfjkaMMSxrOr3bV91rAmpBxWkCISgQ8scTT5pYVE7SQ8RlJxLGG5lxFc4hHNAT7ticbeSw/0?wx_fmt=png",
    "nickname": "微信名称",
    "sex": "1",
    "province": "上海",
    "city": "上海",
    "country": "中国",
    "url_used": "1"
}

def handle_line(line_,f):
    line_ = line_[:-1]
    line = line_.split(",")
    print(line)
    data = dict()
    data['url_used'] = "1"
    data['field'] = "2"
    data['nickname'] = line[0]
    data['sex'] = line[1]
    data['city'] = line[2]
    data['province'] = line[3]
    data['country'] = line[4]
    data['image_url'] = line[5]

    res = requests.post(url=url, headers={"Content-Type": "application/json"}, data=json.dumps(data))
    res_dict = json.loads(res.text)
    code = res_dict.get("code")
    if code != 200:
        print("此次请求该url出错，请求的信息为%s" %str(data))
        f.write("此次请求该url出错，请求的信息为%s" %str(data)+'\n')
        f.write("--" * 30 + '\n')
    else:
        f.write(line_+'\n')
        label = res_dict.get("result")
        for label_ in label:
            li = '{}->{}->{}'.format(label_.get("name"),label_.get("content"),label_.get("score"))
            f.write(li+"\n")
        f.write("--"*30+'\n')


if __name__ == '__main__':
    f = open("data/result_2.txt",'a',encoding='utf8')
    f2 = open("data/data.txt",'r',encoding='utf8')

    for line in f2.readlines():
        handle_line(line,f)