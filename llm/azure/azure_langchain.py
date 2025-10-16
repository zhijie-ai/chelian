# !/user/bin/python
# coding  : utf-8
# @Time   : 2025/1/17 17:54
# @User   : RANCHODZU
# @File   : azure_langchain.py
# @e-mail : zushoujie@ghgame.cn

from openai import AzureOpenAI
from langchain.prompts import PromptTemplate


messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "当前用的是哪个模型，模型版本是啥"},
            ],
        }
    ]

url = f'https://azure-ran-resource-test001.openai.azure.com/'
key = f'9nFYPivGPXUwD9qZO88TPIjMTVh0evDYcoX0h033fEh0mXDos8LdJQQJ99BAACHYHv6XJ3w3AAABACOG2sJo'
url = f'https://azure-ran-resource-test001.openai.azure.com/openai/deployments/gpt-4o-2024-08-06-ft-ca7c656f6e5f41d392fb1c5577a4e19f/chat/completions?api-version=2024-08-01-preview'
client = AzureOpenAI(
    azure_endpoint=url,
    api_key=key,
    api_version="2024-08-01-preview"  # This API version or later is required to access seed/events/checkpoint features
)

# model参数传递的事部署列表中的“名称”而非“模型名称”，通过resource的url可以验证
res = client.chat.completions.create(model=f"gpt-4o-2024-08-06-ft-ca7c656f6e5f41d392fb1c5577a4e19f", messages=messages)
print(res.choices[0].message.content)