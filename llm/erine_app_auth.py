# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/7/24 17:32
# @User   : RANCHODZU
# @File   : erine_app_auth.py
# @e-mail : zushoujie@ghgame.cn
import requests
import json


import requests
import json
import numpy as np
import cv2
import base64
import io
from PIL import Image


def image2base64(img, quality=20):
    img = img.convert('L')
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# 使用的是应用ak/sk
API_KEY = ""
SECRET_KEY = ""

path = 'F:/ocr_data/imgs/av1902895992/av1902895992_48.png'
# 具体走的access_token鉴权方式还是安全认证AK/SK的鉴权方式，其实看传的是QIANFAN_AK还是QIANFAN_ACCESS_KEY。
def main():

    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token=" + get_access_token()

    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": "帮我写首诗"
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
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


def sdk_call():
    import os
    import qianfan
    os.environ["QIANFAN_AK"] = API_KEY
    os.environ["QIANFAN_SK"] = SECRET_KEY
    # os.environ["QIANFAN_ACCESS_KEY"] = '783348ec34464fc885ca804c63f1b77f'
    # os.environ["QIANFAN_SECRET_KEY"] = '4e8d160937c24c42971524e9bd927c23'

    resp = qianfan.ChatCompletion().do(endpoint="completions", temperature=0.95, top_p=0.8, penalty_score=1,
                                       disable_search=False, enable_citation=False, messages=[{
        "role": "user",
        "content": "你好"
    }])

    print(resp.body)

def sdk_call_2():
    import os
    import qianfan
    os.environ["QIANFAN_AK"] = API_KEY
    os.environ["QIANFAN_SK"] = SECRET_KEY
    # os.environ["QIANFAN_ACCESS_KEY"] = '783348ec34464fc885ca804c63f1b77f'
    # os.environ["QIANFAN_SECRET_KEY"] = '4e8d160937c24c42971524e9bd927c23'

    resp = qianfan.Image2Text().do(prompt='这张图片有字幕吗？', image=image2base64(Image.open(path)))

    print(resp.body)


if __name__ == '__main__':
    from qianfan.resources.llm.chat_completion import _ChatCompletionV1
    main()
    # sdk_call()