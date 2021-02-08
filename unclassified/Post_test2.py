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

import json
import requests
import urllib.request as request
import base64
import io

app_id = "623896511427641344"
app_key = "277b42a2be4244a3aa4451520afc8ad6"
fee_type = 0

url = "http://api.unidt.com/api/fractal/openid/report?app_id=%s&app_key=%s&fee_type=0" % (app_id, app_key)

#参数传的是URL
data = {
    # 头像不能为空
    "image_url": "http://mmbiz.qpic.cn/mmbiz_png/UtWdDgynLdbIjl7VGqfjkaMMSxrOr3bV91rAmpBxWkCISgQ8scTT5pYVE7SQ8RlJxLGG5lxFc4hHNAT7ticbeSw/0?wx_fmt=png",
    "nickname": "团团",#可以为空
    "sex": "1",#不能为空
    "city": "Dalian",#可以为空
    "province": "Liaoning",#可以为空
    "country": "China",#可以为空
    "url_used": "1"
}

# base64编码
image_url = "http://mmbiz.qpic.cn/mmbiz_png/UtWdDgynLdbIjl7VGqfjkaMMSxrOr3bV91rAmpBxWkCISgQ8scTT5pYVE7SQ8RlJxLGG5lxFc4hHNAT7ticbeSw/0?wx_fmt=png"
data2 = {
    # "image_url": image_url,
    "nickname": "团团",#可以为空
    "sex": "1",#不能为空
    "city": "Dalian",#可以为空
    "province": "Liaoning",#可以为空
    "country": "China",#可以为空
    "url_used": "0"
}
if __name__ == '__main__':
    # res = requests.post(url=url, headers={"Content-Type": "application/json"}, data=json.dumps(data))
    # print(res.text)
    #
    origin_file = io.BytesIO(request.urlopen(image_url).read())
    base64_str = base64.b64encode(origin_file.getvalue())
    data2['image'] = str(base64_str)[1:]
    print(str(base64_str)[1:])

    res = requests.post(url=url, headers={"Content-Type": "application/json"}, data=json.dumps(data2))
    print(res.text)
