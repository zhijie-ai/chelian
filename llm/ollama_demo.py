# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2026/5/26 16:25                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import requests

url = "http://localhost:11434/api/generate"
data = {
    "model": "qwen2.5:7b",
    "prompt": "今天是多少号",
    "stream": False
}
res = requests.post(url, json=data)
print(res.json()["response"])
