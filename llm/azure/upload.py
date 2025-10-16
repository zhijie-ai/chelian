# !/user/bin/python
# coding  : utf-8
# @Time   : 2025/1/7 15:13
# @User   : RANCHODZU
# @File   : upload.py
# @e-mail : zushoujie@ghgame.cn
from openai import AzureOpenAI

url = f'https://llm-resource-ran.openai.azure.com/'
key = f'5fU5U2qtESCX7q2u0y3i1XAnVXW4kig5QjbkHzNWeyM6AbVfANiKJQQJ99ALACYeBjFXJ3w3AAABACOGOLDU'

client = AzureOpenAI(
  azure_endpoint=url,
  api_key=key,
  api_version="2024-08-01-preview"  # This API version or later is required to access seed/events/checkpoint features
)

training_file_name = 'demo.jsonl'
training_response = client.files.create(
    file=open(training_file_name, "rb"), purpose="fine-tune"
)
training_file_id = training_response.id