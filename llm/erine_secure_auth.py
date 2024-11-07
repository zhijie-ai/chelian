# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/7/24 17:32
# @User   : RANCHODZU
# @File   : erine_app_auth.py
# @e-mail : zushoujie@ghgame.cn

import requests
import json
import numpy as np
import cv2
from PIL import Image

path = 'F:/ocr_data/imgs/3x2sikyhmc7didy/3x2sikyhmc7didy_5.png'
def main():

    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions"

    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": "写一首五言绝句"
            }
        ],
        "temperature": 0.95,
        "top_p": 0.8,
        "penalty_score": 1,
        "disable_search": False,
        "enable_citation": False,
        "response_format": "text"
    })
    headers = {
        'Content-Type': 'application/json',
        # 这是通过百度云上的工具生成的字符串 https://cloud.baidu.com/signature/index.html
        'Authorization': 'bce-auth-v1/783348ec34464fc885ca804c63f1b77f/2024-08-20T10:17:50Z/30000/host/ee709da54e36fc46a09be04c058ec989c119dcf992c9ad27ecf8020977eceda5'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)


def sdk_call():
    import qianfan

    chat_comp = qianfan.ChatCompletion()

    # 指定特定模型
    resp = chat_comp.do(model="ERNIE-3.5-8K", messages=[{
        "role": "user",
        "content": "帮我写一首诗"
    }])

    print(resp["body"])


if __name__ == '__main__':
    # main()
    # print(get_access_token())
    sdk_call()
