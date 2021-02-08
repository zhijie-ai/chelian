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

header={'User-Agent':
            'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24',
        "Content-Type":"application/json"}
header2={'User-Agent':
            'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24',
        "Content-Type":"x-www-form-urlencoded"}

data = {"image":"base64",
        "image_url":"https://wx.qlogo.cn/mmopen/vi_32/zAMHRbOiccjZX9ZcsZQY0WzsVNkhrwslAwcTknJeoeDR9sFa1Tp6tdgDiaPu1goosXzSseJjKianxiaKxkqKjYUWwQ/132",
        "nickname":"田广裕",
        "sex":"1",
        "city":"朔州",
        "province":"山西",
        "country":"中国",
        "url_used":"1"}
url = "http://api.unidt.com/api/fractal/openid/report?app_id=619125947727085568&app_key=5a85862f532c43d0b089c81b4c296a07&fee_type=0"

print(type(json.loads(json.dumps(data))))

r =requests.post(url,json.dumps(data),headers=header)
# print(r.headers)
# print(r.text)
print(r.content)
# print(type(r.text))#<class 'str'>
# print(json.dumps(data))
# print(data)
# print(json.loads(json.dumps(data)))
# print(json.loads(r.text))
# print(type(json.loads(r.text)))#<class 'dict'>
# print(type(json.loads(r.text).get("code")))#<class 'int'>
# print(json.loads(r.text).get("code"))