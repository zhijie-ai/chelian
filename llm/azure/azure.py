# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/12/30 14:15
# @User   : RANCHODZU
# @File   : azure.py
# @e-mail : zushoujie@ghgame.cn

import os
from openai import AzureOpenAI

url = f'https://azure-ran-resource-test001.openai.azure.com/'
key = f'9nFYPivGPXUwD9qZO88TPIjMTVh0evDYcoX0h033fEh0mXDos8LdJQQJ99BAACHYHv6XJ3w3AAABACOG2sJo'
# url = f'https://lightspeed-east-us2.openai.azure.com/'
# key = f'f933dfbc32da47adb414645352f00c2e'
client = AzureOpenAI(
    azure_endpoint=url,
    api_key=key,
    api_version="2024-08-01-preview"  # This API version or later is required to access seed/events/checkpoint features
)
# training_response = client.files.create(
#     file=open('demo.jsonl', "rb"), purpose="fine-tune"
# )
# print(training_response)
# print(client.files.list())

# gpt-4-turbo的部署类型是标准,gpt-4o的类型是全局标准
response = client.chat.completions.create(
  model="gpt-4.1",  # replace with the model deployment name of your o1-preview, or o1-mini model
    messages=[
        {"role": "user", "content": "当前用的是哪个模型"},
    ],
)
res = response.model_dump_json(indent=2)
print(res)
