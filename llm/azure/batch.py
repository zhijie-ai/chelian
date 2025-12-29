# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/12/30 14:15
# @User   : RANCHODZU
# @File   : azure.py
# @e-mail : zushoujie@ghgame.cn

import time
from openai import AzureOpenAI

url = f'https://azure-ran-resource-test001.openai.azure.com/'
key = f''

client = AzureOpenAI(
    azure_endpoint=url,
    api_key=key,
    api_version="2024-08-01-preview"  # This API version or later is required to access seed/events/checkpoint features
)
training_response = client.files.create(
    file=open('batch.jsonl', "rb"), purpose="batch"
)

training_file_id = training_response.id
print("Training file ID:", training_file_id)

try:
    time.sleep(5)
except Exception as e:
    print(e)

# gpt-4-turbo的部署类型是标准,gpt-4o的类型是全局标准

batch = client.batches.create(
    input_file_id=training_file_id,
    endpoint='/v1/chat/completions',
    completion_window="24h",
    metadata={
        "description": "nightly eval job"
    }
)
print(batch.model_dump_json(indent=2))